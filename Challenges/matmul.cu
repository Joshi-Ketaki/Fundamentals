#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 32
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

    
    // Naive - not so great perf wise. 
    /*int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < N)
    {
        float sum = 0.0f;    
        for(int k = 0; k < K; k++)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }*/

    // Tiled

    // +1 as padding to avoid bank conflicts
    extern __shared__ float shared_mem[];
	float* As = shared_mem;
	float* Bs = shared_mem + (TILE_SIZE * TILE_SIZE);
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float zero = 0.0f; //__float2half(0.0f);
    float sum = 0.0f;

    // Loop over tile along the K dimension
    for (int t = 0; t < (K + TILE_SIZE -1)/TILE_SIZE; t++)
    {
        // Load tile into shared memory
        int Acol = t * TILE_SIZE + threadIdx.x;
        int Brow = t * TILE_SIZE + threadIdx.y;
        As[threadIdx.y * TILE_SIZE + threadIdx.x] = (row < M && Acol < K) ? A[row * K + Acol]: zero;
        Bs[threadIdx.y * TILE_SIZE + threadIdx.x] = (Brow < K && col < N) ? B[Brow * N + col]: zero;
        __syncthreads();

        for (int k = 0; k < min(TILE_SIZE, K - (t*TILE_SIZE)); k++)
        {
            sum += (As[threadIdx.y * TILE_SIZE + k]) * (Bs[k * TILE_SIZE + threadIdx.x]);
        }
        __syncthreads();
    }
    if(row < M && col < N)
        C[row * N + col] = sum;

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {

    if (M < 1 || N < 1 || K < 1 || M > 8192 || N > 8192 || K > 8192) return;


    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE); // 16,16
    // dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
    //                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dim3 blocksPerGrid((N+TILE_SIZE-1)/TILE_SIZE,
                        (M+TILE_SIZE-1)/TILE_SIZE);

    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
