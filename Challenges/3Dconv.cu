#include <cuda_runtime.h>

// input [input_Depth][input_ht][input_wd]
// output [output_depth][output_ht][output_wd]
// kernel [kernel_depthl][KERNEL_SIZE][KERNEL_SIZE]

__global__ void conv3d(const float *d_input, const float *d_kernel, float *d_output, 
                   int INPUT_HT, int INPUT_WD, int KERNEL_HT, int KERNEL_WD, int OUTPUT_HT, int OUTPUT_WD,
                   int input_depth, int kernel_depth, int output_depth)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_z = blockIdx.z * blockDim.z + threadIdx.z;

    float sum = 0.0f;
    if(out_x >= OUTPUT_WD || out_y >= OUTPUT_HT || out_z >= output_depth) 
        return;
    for(int c = 0; c < kernel_depth; c++)
    {
        for(int m = 0; m < KERNEL_HT; m++)
        {
            for(int n = 0; n < KERNEL_WD; n++)
            {
                int inp_depth = out_z + c;
                int inp_ht = out_y + m;
                int inp_wd = out_x + n;
                int inp_idx = (inp_depth * INPUT_HT + inp_ht) * INPUT_WD + inp_wd;
                int ker_idx = (c * KERNEL_HT + m) * KERNEL_WD + n;
                sum += d_input[inp_idx] * d_kernel[ker_idx];
            }
        }
    }
    int op_idx = (out_z * OUTPUT_HT + out_y) * OUTPUT_WD + out_x;
    d_output[op_idx] = sum;
}


// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_depth, int input_rows, int input_cols,
           int kernel_depth, int kernel_rows, int kernel_cols) {
 
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    int output_depth = input_depth - kernel_depth + 1;
    dim3 blockSize(8, 8, 4);
    dim3 gridSize((output_cols + blockSize.x - 1) / blockSize.x, 
                  (output_rows + blockSize.y - 1) / blockSize.y, 
                  (output_depth + blockSize.z - 1) / blockSize.z);

    // Launch the convolution kernel
    conv3d<<<gridSize, blockSize>>>(input, kernel, output, 
                                    input_rows, input_cols, kernel_rows, kernel_cols, output_rows, output_cols,
                                    input_depth, kernel_depth, output_depth);

}
 
