#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int mirrorIdx = N - idx - 1;

    if(idx < N/2)
    {
        float temp = input[idx];
        input[idx] = input[mirrorIdx];
        input[mirrorIdx] = temp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
