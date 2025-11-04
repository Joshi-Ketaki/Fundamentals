#include<stdio.h>
#include<math.h>


void initMat(float* A, int size)
{
    for(int i = 0; i < size; i++)
    {
        A[i] = rand() % 10;
    }
}

void initMatConstantVal(float* A, int size, float val)
{
    for(int i = 0; i < size; i++)
    {
        A[i] = val;
    }
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

void layerNorm_cpu(float* X, float* Y, float* G, float* B, int batch_size, int feature_size)
{
    float var, epsilon = 1e-5;
    for(int batch = 0; batch < batch_size; batch++)
    {
        // calculate mean
        float mean = 0.0f;
        for(int feature = 0; feature < feature_size; feature++)
        {
            mean += X[batch * feature_size + feature];
        }
        mean = mean / feature_size;

        // calculate variation
        float sum_diff = 0.0f, diff;
        for(int feature = 0; feature < feature_size; feature++)
        {
            diff = X[batch * feature_size + feature] - mean;
            sum_diff += (diff * diff);
        }
        var = sum_diff / feature_size;

        //calculate x_hat
        for(int feature = 0; feature < feature_size; feature++)
        {
            float x_hat = (X[batch * feature_size + feature] - mean) / sqrtf(var + epsilon);
            float g = G ? G[feature] : 1.0f;
            float b = B ? B[feature] : 0.0f;
            Y[batch * feature_size + feature] = x_hat * g + b;
        }
    }
}

void calcMeanAndVar(float* X, float* allMean, float* allVar, int batch_size, int feature_size)
{
    for(int batch = 0; batch < batch_size; batch++)
    {
        // calculate mean
        float mean = 0.0f;
        for(int feature = 0; feature < feature_size; feature++)
        {
            mean += X[batch * feature_size + feature];
        }
        allMean[batch] = mean / feature_size;

        // calculate variation
        float sum_diff = 0.0f, diff;
        for(int feature = 0; feature < feature_size; feature++)
        {
            diff = X[batch * feature_size + feature] - allMean[batch];
            sum_diff += (diff * diff);
        }
        allVar[batch] = sum_diff / feature_size;
    }
}

__global__ void layerNorm_kernel(float* X, float* Y, float* G, float* B, float* allMean, float* allVar, int batch_size, int feature_size)
{
    int batch = blockDim.y * blockIdx.y + threadIdx.y;
    int feature = blockDim.x * blockIdx.x + threadIdx.x;
    float epsilon = 1e-5;
    if(batch < batch_size && feature < feature_size)
    {
        float x_hat = (X[batch * feature_size + feature] - allMean[batch]) / sqrtf(allVar[batch] + epsilon);
        float g = G ? G[feature] : 1.0f;
        float b = B ? B[feature] : 0.0f;
        Y[batch * feature_size + feature] = x_hat * g + b;
    }

}

__global__ void layerNorm_optimized_kernel(float* X, float*Y, float*G, float*B, int batch_size, int feature_size)
{
    extern __shared__ float shared_mem[];
    float* temp = shared_mem;

    int batch = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float epsilon = 1e-5;
    // calc sum
    float sum = 0.0f;
    for(int i = tid; i < feature_size; i += blockDim.x)
    {
        sum += X[batch * feature_size + i];
    }
    if (tid < blockDim.x) temp[tid] = (tid < feature_size) ? sum : 0.0f;

    // reduction and mean 
    int activeThreads = 1;
    while (activeThreads < feature_size) activeThreads <<= 1;
    for(int i = activeThreads/2; i > 0; i >>= 1)
    {
        if (tid < i)
            temp[tid] += temp[tid + i];
        __syncthreads();

    }
    float mean = temp[0] / feature_size;

    // calc var
    float diff = 0.0f;
    sum = 0.0f;
    for(int i = tid; i < feature_size; i+=stride)
    {
        diff = X[batch * feature_size + i] - mean;
        sum += diff * diff;
    }
    if (tid < blockDim.x) temp[tid] = (tid < feature_size) ? sum : 0.0f;

    //reduction and mean
    activeThreads = 1;
    while (activeThreads < feature_size) activeThreads <<= 1;
    for(int i = activeThreads/2; i > 0; i >>=1)
    {
        if (tid < i)
            temp[tid] += temp[tid + i]; 
        __syncthreads();
    }
    float var = temp[0] / feature_size;
    
    if (tid >= feature_size) return;
    for(int i = tid; i < feature_size; i+=stride)
    {
        float x_hat = (X[batch * feature_size + i] - mean) / (sqrtf(var + epsilon));
        float g = G ? G[i] : 1.0f;
        float b = B ? B[i] : 0.0f;
        Y[batch * feature_size + i] = x_hat * g + b;
    }
}

int main()
{
    // say batch size is 5 and feature size is 4
    // because leetgpu will not allow command line arg
    int batch_size = 5, feature_size = 1024;
    int matSize = batch_size * feature_size;
    float *X = (float*)malloc(sizeof(float) * matSize);
    float *cpu_Y = (float*)malloc(sizeof(float) * matSize);
    float *gpu_Y = (float*)malloc(sizeof(float) * matSize);
    float *G = (float*)malloc(sizeof(float) * feature_size);
    float *B = (float*)malloc(sizeof(float) * feature_size);

    initMat(X, matSize);
    initMatConstantVal(G, feature_size, 1.0f);
    initMatConstantVal(B, feature_size, 0.0f);
    
    layerNorm_cpu(X, cpu_Y, G, B, batch_size, feature_size);

    float *allMean = (float*)malloc(sizeof(float) * batch_size);
    float *allVar = (float*)malloc(sizeof(float) * batch_size);
    calcMeanAndVar(X, allMean, allVar, batch_size, feature_size);

    float *d_X, *d_Y, *d_G, *d_B, *d_allMean, *d_allVar;
    cudaMalloc(&d_X, sizeof(float) * matSize);
    cudaMalloc(&d_Y, sizeof(float) * matSize);
    cudaMalloc(&d_G, sizeof(float) * feature_size);
    cudaMalloc(&d_B, sizeof(float) * feature_size);
    cudaMalloc(&d_allMean, sizeof(float) * batch_size);
    cudaMalloc(&d_allVar, sizeof(float) * batch_size);

    cudaMemcpy(d_X, X, sizeof(float) * matSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_G, G, sizeof(float) * feature_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * feature_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_allMean, allMean, sizeof(float) * batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_allVar, allVar, sizeof(float) * batch_size, cudaMemcpyHostToDevice);

    //dim3 blockSize(256, 1, 1);
    //dim3 gridSize((feature_size + blockSize.x - 1)/blockSize.x,
    //              (batch_size + blockSize.y - 1)/blockSize.y, 1);
    int blockSize = feature_size > 1024 ? 256 : feature_size;
    int gridSize = batch_size;
    size_t sharedMemSize = blockSize * sizeof(float);
    layerNorm_optimized_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_X, d_Y, d_G, d_B, batch_size, feature_size);
    cudaMemcpy(gpu_Y, d_Y, sizeof(float) * matSize, cudaMemcpyDeviceToHost);
    validate_outputs(cpu_Y, gpu_Y, 5);

    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_G);
    cudaFree(d_B);
    cudaFree(d_allMean);
    cudaFree(d_allVar);

    free(X);
    free(cpu_Y);
    free(gpu_Y);
    free(G);
    free(B);
    free(allMean);
    free(allVar);

    return 0;
}