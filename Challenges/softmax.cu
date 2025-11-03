#include <cuda_runtime.h>

__global__ void softmax_kernel_partial_max(const float *d_input, float *d_blockMax, int N)
{
    extern __shared__ float sh_mem[];
    float* sh_input = sh_mem;
    float *block_max = sh_mem + blockDim.x;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    //Load the input into shared memory
    if(idx < N)
        sh_input[threadIdx.x] = d_input[idx];
    else
        sh_input[threadIdx.x] = -INFINITY; // Handle out of bounds else the remaining threads in that block will write garbage. -INFINITY will not mess up max calculations. 
    
    // Compute the max value in the block
    if(threadIdx.x == 0)
    {
        float max_val = sh_input[0];
        // bounds should be within the block && within N -- a block could go beyond N. think 1025
        for(int i = 0; i < blockDim.x && blockDim.x * blockIdx.x + i < N; i++)
        {
            if(sh_input[i] > max_val)
                max_val = sh_input[i];
        }
        *block_max = max_val;
    }
    __syncthreads(); 
    // Copy to global memory
   if (threadIdx.x == 0) d_blockMax[blockIdx.x] = *block_max;

}
float softmax_global_max(float* blockMax, int numBlocks)
{
    float max_val = blockMax[0];
    // bounds should be within the block && within N -- a block could go beyond N. think 1025
    for(int i = 0; i < numBlocks; i++)
    {
        if(blockMax[i] > max_val)
            max_val = blockMax[i];
    }
    // Copy to global memory
    return max_val;
}

__global__ void softmax_kernel_partial_exp(const float*d_input, float *globalMax, float* d_output, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N)
        d_output[idx] = expf(d_input[idx] - *globalMax);
}

__global__ void softmax_normalize(float* output, int N, float* sum)
{
    //float sum = 0.0f;
    //for (int i = 0; i < N; i++)
    //    sum += output[i];
    //for (int i = 0; i < N; i++)
    //    output[i] /= sum;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N)
        output[idx] = output[idx]/ *sum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* d_input, float* d_output, int N) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    float max_val, *temp;
    float *gpu_output = (float*) malloc(sizeof(float) * N);
    float *d_blockMax, *globalMax;
    cudaMalloc(&d_blockMax, sizeof(float) * gridSize);
    cudaMalloc(&globalMax, sizeof(float));
    cudaMalloc(&temp, sizeof(float));
    float *blockMax = (float*) malloc(sizeof(float) * gridSize);
    float *blockSum = (float*) malloc(sizeof(float) * gridSize);
    for (int i = 0; i < gridSize; i++) blockSum[i] = 0.0f;
    //softmax_kernel_simple<<<1, N, (N + 2) * sizeof(float)>>>(d_input, d_output, N);
    softmax_kernel_partial_max<<<gridSize, blockSize, sizeof(float) * (blockSize + 1)>>>(d_input, d_blockMax, N);
    cudaDeviceSynchronize();
    cudaMemcpy(blockMax, d_blockMax, sizeof(float) * gridSize, cudaMemcpyDeviceToHost);
    max_val = softmax_global_max(blockMax, gridSize);
    cudaMemcpy(globalMax, &max_val, sizeof(float), cudaMemcpyHostToDevice);
    softmax_kernel_partial_exp<<<gridSize, blockSize, sizeof(float) * (blockSize + 1)>>>(d_input, globalMax, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_output, d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);
    float sum = 0.0f;
    for(int i = 0; i < N; i++) 
        sum += gpu_output[i];
    cudaMemcpy(temp, &sum, sizeof(float), cudaMemcpyHostToDevice);
    softmax_normalize<<<gridSize, blockSize>>>(d_output, N, temp);
    cudaDeviceSynchronize();
}
