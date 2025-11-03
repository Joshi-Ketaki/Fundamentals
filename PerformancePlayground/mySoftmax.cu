#include<stdio.h>
#include<math.h>

float find_max(float *input, int N)
{
    float max_val = input[0];
    for(int i = 1; i < N; i++)
    {
        if(input[i] > max_val)
            max_val = input[i];
    }
    return max_val;
}

void validate_outputs(float *cpu_output, float* gpu_output, int num_samples)
{
    printf("\nValidating first %d output elements...\n", num_samples);
    for (int s = 0; s < num_samples; s++)
    {
        int idx = rand() % num_samples;
        if(fabs(cpu_output[idx] - gpu_output[idx]) > 1e-5)
        {
            printf("\n Output mismatch!");
            return;
        }
        printf("\n Outputs match");
        printf("\n CPU Output[%d] = %.3f\n", idx,  cpu_output[idx]);
        printf("\n GPU Output[%d] = %.3f\n", idx, gpu_output[idx]);
    }
    printf("\n(If these values look finite and not NaN/Inf, indexing is likely correct.)\n");
}

void softmax_cpu(float *input, float *output, int N, float max_val)
{
    //Compute the exponentials and also the sum
    float sum = 0.0f;
    for(int i = 0; i < N; i++)
    {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Normalize the output
    for(int i = 0; i < N; i++)
    {
        output[i] = output[i] / sum;
    }
}

__global__ void softmax_kernel_simple(float* d_input, float* d_output, int N)
{
    int idx = threadIdx.x;
    // Load inpt into shared memory
    extern __shared__ float sh_mem[];
    float *sh_input = sh_mem;
    float *max_val = sh_mem + N;
    float *sum = sh_mem + N + 1;

    if(idx < N)
        sh_input[idx] = d_input[idx];

    __syncthreads();

    // Compute the max value
    if (threadIdx.x == 0)
    {
        *max_val = sh_input[0];
        for(int i = 0; i < N; i++)
        {
            if(sh_input[i] > *max_val)
                *max_val = sh_input[i];
        }
    }
    __syncthreads();

    // Compute exponentials
    float e = expf(sh_input[idx] - *max_val);
    if(idx < N)
        sh_input[idx] = e;

    __syncthreads();

    //Compute sum
    if(threadIdx.x == 0)
    {
        *sum = 0.0f;
        for(int i = 0; i < N; i++)
        {
            *sum += sh_input[i];
        }
    }
    __syncthreads();

    //Normalize
    if(idx < N)
        d_output[idx] = sh_input[idx] / *sum;
    
}

__global__ void softmax_kernel_partial_max(float *d_input, float *d_blockMax, int N)
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
__global__ void softmax_kernel_global_max(float* d_blockMax, float* globalMax, int numBlocks)
{
    // Compute the max value in the block
    if(threadIdx.x == 0)
    {
        float max_val = d_blockMax[0];
        // bounds should be within the block && within N -- a block could go beyond N. think 1025
        for(int i = 0; i < numBlocks; i++)
        {
            if(d_blockMax[i] > max_val)
                max_val = d_blockMax[i];
        }
        // Copy to global memory
        *globalMax = max_val;
    } 
}

__global__ void softmax_kernel_partial_exp(float*d_input, float *globalMax, float* d_output, int N)
{
    extern __shared__ float sh_mem[];
    float* sh_input = sh_mem;
    float* gMax = sh_mem + blockDim.x;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Load the input
    if(idx < N)
        sh_input[threadIdx.x] = d_input[idx];
    else
        sh_input[threadIdx.x] = -INFINITY;
    
    if(threadIdx.x == 0)
        *gMax = *globalMax;
    __syncthreads();


    // Calculate the exponential
    if(idx < N)
        d_output[idx] = expf(sh_input[threadIdx.x] - *gMax);
}

__global__ void softmax_kernel_normalize(float* d_output, int N)
{
    float sum = 0.0f;
    for (int i = 0; i < N; i++)
        sum += d_output[i];
    for (int i = 0; i < N; i++)
        d_output[i] /= sum;
}

int main()
{
    int N = 4;
    float max_val;
    float *input = (float*) malloc(sizeof(float) * N);
    float *cpu_output = (float*) malloc(sizeof(float) * N);
    float *gpu_output = (float*) malloc(sizeof(float) * N);
    // Initialize input with some values
    for(int i = 0; i < N; i++)
    {
        input[i] = 1.0f * (rand() % 10);
    }

    // Find max value in input
    max_val = find_max(input, N);
    // Compute softmax on CPU
    softmax_cpu(input, cpu_output, N, max_val);

    // Compute softmax on GPU
    float *d_input, *d_output, *d_blockMax, *globalMax;
    int blockSize = 256;
    int gridSize = (N + blockSize - 1)/blockSize;
    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * N);
    cudaMalloc(&d_blockMax, sizeof(float) * gridSize);
    cudaMalloc(&globalMax, sizeof(float));

    cudaMemcpy(d_input, input, sizeof(float) * N, cudaMemcpyHostToDevice);
    //softmax_kernel_simple<<<1, N, (N + 2) * sizeof(float)>>>(d_input, d_output, N);
    softmax_kernel_partial_max<<<gridSize, blockSize, sizeof(float) * (blockSize + 1)>>>(d_input, d_blockMax, N);
    softmax_kernel_global_max<<<1, 1>>>(d_blockMax, globalMax, gridSize);
    softmax_kernel_partial_exp<<<gridSize, blockSize, sizeof(float) * (blockSize + 1)>>>(d_input, globalMax, d_output, N);
    softmax_kernel_normalize<<<1, 1>>>(d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_output, d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Validate outputs
    validate_outputs(cpu_output, gpu_output, N);

    free(input);
    free(cpu_output);
    free(gpu_output);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blockMax); 
    cudaFree(globalMax);
    return 0;
}