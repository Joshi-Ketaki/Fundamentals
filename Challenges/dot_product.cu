#include <cuda_runtime.h>

__global__ void dotProd_kernel2(const float* A, const float* B, int N, float* result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        float local_mul = A[idx] * B[idx];
        atomicAdd(result, local_mul);
    }
}

__global__ void dotProd_kernel(const float* A, const float* B, int N, float* result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Load input into shm
    extern __shared__ float shmem[];
    float *prod = shmem;

    float sum = 0.0f;
    
    for(int i = idx; i < N; i += blockDim.x * gridDim.x)
    {
       sum += A[i] * B[i];
    }
    prod[threadIdx.x] = sum;
    __syncthreads();

    for(int i = blockDim.x/2; i > 0; i>>=1)
    {
        if(threadIdx.x < i)
            prod[threadIdx.x] += prod[threadIdx.x + i];
        __syncthreads();
    }
    if(threadIdx.x == 0)
        atomicAdd(result, prod[0]);
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((N + blockSize.x- 1)/blockSize.x, 1, 1);
    int shmemSize = sizeof(float) * blockSize.x;
    dotProd_kernel2<<<gridSize, blockSize, shmemSize>>>(A, B, N, result);
    cudaDeviceSynchronize();
}
