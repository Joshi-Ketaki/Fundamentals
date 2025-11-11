#include <cuda_runtime.h>
#define TILE_SIZE 32

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;

    extern __shared__ float shmem[];
    float* As = shmem;
    float* Bs = shmem + (TILE_SIZE * TILE_SIZE);

    float sum = 0.0f;
    for(int t = 0; t < (N + TILE_SIZE - 1)/ TILE_SIZE; t++)
    {
        // Load input A into shared memory
        if(row < M && (t * TILE_SIZE + threadIdx.x) < K)
            As[threadIdx.y * TILE_SIZE + threadIdx.x] = A[batch * M * K +  row * K + (t * TILE_SIZE + threadIdx.x)]; 
        else
            As[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        // Load input B from shared memory
        if((t * TILE_SIZE + threadIdx.y) < K && col < N)
            Bs[threadIdx.y * TILE_SIZE + threadIdx.x] = B[batch * N * K + (t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        __syncthreads();
        // Multiply and add
        for(int n = 0; n < TILE_SIZE; n++)
        {
            sum += As[threadIdx.y * TILE_SIZE + n] * Bs[n * TILE_SIZE + threadIdx.x];
        }
        __syncthreads();
    }
    if(row < M && col < N)
    {
        C[batch * M * N + row * N + col] = sum;
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((K + TILE_SIZE - 1) / TILE_SIZE,
                       (M + TILE_SIZE - 1) / TILE_SIZE, BATCH);
    size_t sharedMemSize = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(A, B, C, BATCH, M, N, K);
    cudaDeviceSynchronize();
} 
