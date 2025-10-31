#include <iostream>
#include <cuda_runtime.h>
#include <ctime>
using namespace std;

int C_in = 2, C_out = 1;
int INPUT_HT = 16, INPUT_WD = 16, KERNEL_SIZE = 3;
int OUTPUT_HT = INPUT_HT - KERNEL_SIZE + 1;
int OUTPUT_WD = INPUT_WD - KERNEL_SIZE + 1;

// --------------------------------------------------------------
// Helper function to print a slice of tensor for debugging
// --------------------------------------------------------------
void print_tensor(float *data, int C, int H, int W, const char *name)
{
    printf("\nTensor: %s\n", name);
    for (int c = 0; c < C; c++)
    {
        printf("Channel %d:\n", c);
        for (int i = 0; i < H; i++)
        {
            for (int j = 0; j < W; j++)
            {
                int idx = (c * H + i) * W + j;
                printf("%6.1f ", data[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

// --------------------------------------------------------------
// Validate a few output elements manually
// --------------------------------------------------------------
void validate_outputs(float *cpu_output, float* gpu_output, int num_samples)
{
    printf("\nValidating first %d output elements...\n", num_samples);
    for (int s = 0; s < num_samples; s++)
    {
        int ch = rand() % C_out;
        int i = rand() % OUTPUT_HT;
        int j = rand() % OUTPUT_WD;
        int idx = (ch * OUTPUT_HT + i) * OUTPUT_WD + j;
        if(cpu_output[idx] != gpu_output[idx])
        {
            printf("\n Output mismatch!");
            return;
        }
        printf("\n Outputs match");
        printf("\n CPU Output[%d][%d][%d] = %.3f\n", ch, i, j, cpu_output[idx]);
        printf("\n GPU Output[%d][%d][%d] = %.3f\n", ch, i, j, gpu_output[idx]);
    }
    printf("\n(If these values look finite and not NaN/Inf, indexing is likely correct.)\n");
}

void conv2d_cpu(float *input, float *kernel, float *output)
{
    for(int ch = 0; ch < C_out; ch++)
    {
        for(int i = 0; i < OUTPUT_HT; i++)
        {
            for(int j = 0; j < OUTPUT_WD; j++)
            {
                float sum = 0.0f;
                for(int c = 0; c < C_in; c++)
                {
                    for(int m = 0; m < KERNEL_SIZE; m++)
                    {
                        for(int n = 0; n < KERNEL_SIZE; n++)
                        {
                            int inp_idx = (c * INPUT_HT + ( i + m)) * INPUT_WD + (j + n);
                            int ker_idx = ((ch * C_in + c) * KERNEL_SIZE + m) * KERNEL_SIZE + n;
                            sum += input[inp_idx] * kernel[ker_idx];
                        }
                    }
                }
                int out_idx = (ch * OUTPUT_HT + i) * OUTPUT_WD + j;
                output[out_idx] = sum;
            }
        }
    }   
}

// input [channel][input_ht][input_wd]
// output [channel][output_ht][output_wd]
// kernel [out_channel][inp_channel][KERNEL_SIZE][KERNEL_SIZE]

__global__ void conv2d(float *input, float* kernel, float* output, 
                       int INPUT_HT, int INPUT_WD, int KERNEL_SIZE, int OUTPUT_HT, int OUTPUT_WD,
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
        for(int m = 0; m < KERNEL_SIZE; m++)
        {
            for(int n = 0; n < KERNEL_SIZE; n++)
            {
                int inp_idx = (c * INPUT_HT + out_y+m) * INPUT_WD + out_x+n;
                int kern_idx = ((out_z * C_in + c) * KERNEL_SIZE + m) * KERNEL_SIZE + n;
                sum += input[inp_idx] * kernel[kern_idx];
            }
        }
    }
    int op_idx = ((out_z * OUTPUT_HT) + out_y) * OUTPUT_WD + out_x;
    output[op_idx] = sum;

}

int main()
{

    float *input = (float*)malloc((size_t)C_in * INPUT_HT * INPUT_WD * sizeof(float));
    float* kernel = (float*)malloc((size_t)C_out * (size_t)C_in * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float* cpu_output = (float*)malloc((size_t)C_out * OUTPUT_HT * OUTPUT_WD * sizeof(float));
    float* gpu_output = (float*)malloc((size_t)C_out * OUTPUT_HT * OUTPUT_WD * sizeof(float));

    // Initialize input and kernel with some values
    for(int i = 0; i < (size_t)C_in * INPUT_HT * INPUT_WD; i++)
    {
        input[i] = i % 10;
    }
    for (int i = 0; i < (size_t)C_out * (size_t)C_in * KERNEL_SIZE * KERNEL_SIZE; i++)
    {
        kernel[i] = i % 10;
    }
    conv2d_cpu(input, kernel, cpu_output);
    //validate_outputs(cpu_output, 5); 
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, (size_t)C_in * INPUT_HT * INPUT_WD * sizeof(float));
    cudaMalloc(&d_kernel, (size_t)C_out * (size_t)C_in * KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_output, (size_t)C_out * OUTPUT_HT * OUTPUT_WD * sizeof(float));

    cudaMemcpy(d_input, input, (size_t)C_in * INPUT_HT * INPUT_WD *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, (size_t)C_out * (size_t)C_in * KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16, 1);
    dim3 gridSize((OUTPUT_WD + blockSize.x - 1) / blockSize.x, 
                  (OUTPUT_HT + blockSize.y - 1) / blockSize.y, C_out);

    // Launch the convolution kernel
    conv2d<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, 
                                    INPUT_HT, INPUT_WD, KERNEL_SIZE, OUTPUT_HT, OUTPUT_WD,
                                    C_in, C_out);
    cudaMemcpy(gpu_output, d_output, (size_t)C_out * OUTPUT_HT * OUTPUT_WD * sizeof(float), cudaMemcpyDeviceToHost);
    validate_outputs(cpu_output, gpu_output, 5); 
    free(input);
    free(kernel);
    free(cpu_output);
    free(gpu_output);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    return 0;
}

