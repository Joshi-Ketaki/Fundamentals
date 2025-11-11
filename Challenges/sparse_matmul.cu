#include <cuda_runtime.h>

#define warp_size 32

__global__ void naive_sparseMatMul(const float* A, const float* x, float* y, int M, int N, int nnz)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;
    if(row < M)
    {
        for(int col = 0; col < N; col++)
        {
            float val = A[row * N + col];
            if(val != 0)
                sum += val * x[col];
        }
        y[row] = sum;
    }
}


__global__ void sparseMatMul(const float* A, const float* x, float* y, int M, int N, int nnz)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for(int i = tid; i < N; i+= blockDim.x)
    {
        sum += A[row * N + i] * x[i];
    }
    for(int i = warp_size/2; i > 0; i >>=1)
    {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
    }
    if(threadIdx.x == 0)
        y[row] = sum;
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {

    //dim3 blockSize(16, 16, 1);
    //dim3 gridSize((N + blockSize.x - 1)/ blockSize.x,
    //               (M + blockSize.y - 1) / blockSize.y, 1);
    int blockSize = warp_size;
    int gridSize = M;

    sparseMatMul<<<gridSize, blockSize>>>(A, x, y, M, N, nnz);
    cudaDeviceSynchronize();
} 
