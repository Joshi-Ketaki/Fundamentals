#include <cuda_runtime.h>

#define TILE_SIZE 32
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    
    // naive
    /*int i_row = blockIdx.y * blockDim.y + threadIdx.y;
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (i_row < rows && i_col < cols)
        output[i_col * rows + i_row] = input[i_row * cols + i_col];*/

    // tiled
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shmem[];

    // load the input
    // coalesced reads from global memory
    int idx = y * cols + x;
    if(x < cols && y < rows)
        shmem[threadIdx.y * TILE_SIZE + threadIdx.x] = input[idx];
    __syncthreads();

    // perform the transpose
    // coalesced write to global memory.
    int out_rows = cols;
    int out_cols = rows;
    int ox = blockIdx.y * TILE_SIZE + threadIdx.x;
    int oy = blockIdx.x * TILE_SIZE + threadIdx.y;
    int oIdx = oy * out_cols + ox;
    if(ox < out_cols && oy < out_rows)
        // Note that we do non coalsced read from shared memory - which is okay
        output[oIdx] = shmem[threadIdx.x * TILE_SIZE + threadIdx.y];
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    if(rows < 1 || rows > 8192 || cols < 1 || cols > 8192) return; 
    float shmemSize = ((TILE_SIZE * TILE_SIZE) * sizeof(float));
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((cols + TILE_SIZE - 1) / TILE_SIZE,
                       (rows + TILE_SIZE - 1) / TILE_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock, shmemSize>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
