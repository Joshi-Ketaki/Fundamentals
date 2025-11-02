#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
        int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
        int output_size = input_size - kernel_size + 1;
        float sum = 0.0f;
        if(outIdx >= output_size) return;
        for(int m = 0; m < kernel_size; m++)
        {
            sum += input[outIdx + m] * kernel[m];
        }
        output[outIdx] = sum;
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    if(input_size < 1 || input_size > 1500000) return;
    if(kernel_size < 1 || kernel_size > 2047) return;

    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
