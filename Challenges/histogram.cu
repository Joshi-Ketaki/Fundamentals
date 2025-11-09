#include <cuda_runtime.h>

__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // think of the case if array size exceeds block size
    // you will need to use reduction in some form when 
    // accumulating the sum
    // so best to use shared memory

    // load input 
    extern __shared__ int shmem[];
    //shm size(num_bins) may be greater than blockdim.x
    for(int i = threadIdx.x; i < num_bins; i+=blockDim.x)
        shmem[i] = 0;

    __syncthreads();

    // total number of elements may be greater than total number of threads
    for(int i = idx; i < N; i += blockDim.x * gridDim.x)
    {
        int data = input[i];
        if(data >= 0 && data < num_bins)
            atomicAdd(&shmem[data], 1);
    }
    __syncthreads();

    // write to output.
    // other threads also must have written so do add
    for(int i = threadIdx.x; i < num_bins; i+=blockDim.x)
    {
        atomicAdd(&histogram[i], shmem[i]);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((N + blockSize.x- 1)/blockSize.x, 1, 1);
    int shmemSize = sizeof(int) * num_bins;
    histogram_kernel<<<gridSize, blockSize, shmemSize>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
