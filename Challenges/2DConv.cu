#include <cuda_runtime.h>


__global__ void conv2d(const float *input, const float* kernel, float* output, 
                       int INPUT_HT, int INPUT_WD, int KERNEL_HT, int KERNEL_WD, int OUTPUT_HT, int OUTPUT_WD,
                    int C_in, int C_out)
{
    // calculate the output indexes
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_z = blockIdx.z;

    if (out_x >= OUTPUT_WD || out_y >= OUTPUT_HT || out_z >= C_out)
        return;

    float sum = 0.0f;

    for(int c = 0; c < C_in; c++)
    {
        for(int m = 0; m < KERNEL_HT; m++)
        {
            for(int n = 0; n < KERNEL_WD; n++)
            {
                int inp_idx = (c * INPUT_HT + out_y+m) * INPUT_WD + out_x+n;
                int kern_idx = ((out_z * C_in + c) * KERNEL_HT + m) * KERNEL_WD + n;
                sum += input[inp_idx] * kernel[kern_idx];
            }
        }
    }
    int op_idx = ((out_z * OUTPUT_HT) + out_y) * OUTPUT_WD + out_x;
    output[op_idx] = sum;

}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {

    if(input_rows < 1 || input_rows > 3072) return;
    if(input_cols < 1 || input_cols > 3072) return;
    if(kernel_rows > input_rows || kernel_cols > input_cols) return;
    
    int output_cols = input_cols - kernel_cols + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int C_in = 1; int C_out = 1;
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((output_cols + blockSize.x - 1) / blockSize.x, 
                  (output_rows + blockSize.y - 1) / blockSize.y, C_out);

    // Launch the convolution kernel
    conv2d<<<gridSize, blockSize>>>(input, kernel, output, 
                                    input_rows, input_cols, kernel_rows, kernel_cols, output_rows, output_cols,
                                    C_in, C_out);

}
