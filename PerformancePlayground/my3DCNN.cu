#include <iostream>
#include <cuda_runtime.h>
#include <ctime>
using namespace std;

int input_depth = 4, kernel_depth = 2;
int INPUT_HT = 16, INPUT_WD = 16, KERNEL_HT = 3, KERNEL_WD = 3;
int OUTPUT_HT = INPUT_HT - KERNEL_HT + 1;
int OUTPUT_WD = INPUT_WD - KERNEL_WD + 1;
int output_depth = input_depth - kernel_depth + 1;

// --------------------------------------------------------------
// Helper functions
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

void validate_outputs(float *cpu_output, int num_samples)
{
    printf("\nValidating first %d output elements...\n", num_samples);
    for (int s = 0; s < num_samples; s++)
    {
        int ch = rand() % output_depth;
        int i = rand() % OUTPUT_HT;
        int j = rand() % OUTPUT_WD;
        int idx = (ch * OUTPUT_HT + i) * OUTPUT_WD + j;
        //if(cpu_output[idx] != gpu_output[idx])
        //{
        //    printf("\n Output mismatch!");
        //    return;
        //}
        //printf("\n Outputs match");
        printf("\n CPU Output[%d][%d][%d] = %.3f\n", ch, i, j, cpu_output[idx]);
        //printf("\n GPU Output[%d][%d][%d] = %.3f\n", ch, i, j, gpu_output[idx]);
    }
    printf("\n(If these values look finite and not NaN/Inf, indexing is likely correct.)\n");
}


void conv3d_cpu(float *input, float *kernel, float* output)
{
    for(int ch = 0; ch < output_depth; ch++)
    {
        for(int i = 0; i < OUTPUT_HT; i++)
        {
            for(int j = 0; j < OUTPUT_WD; j++)
            {
                float sum = 0.0f;
                for (int c = 0; c < kernel_depth; c++)
                {
                    for (int m = 0; m < KERNEL_HT; m++)
                    {
                        for(int n = 0; n < KERNEL_WD; n++)
                        {
                            int inp_depth = ch + c;
                            int inp_ht = i + m;
                            int inp_wd = j + n;;
                            int inp_idx = (inp_depth * INPUT_HT + inp_ht) * INPUT_WD + inp_wd;
                            int ker_idx =  (c * KERNEL_HT + m) * KERNEL_WD + n;
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

int main()
{

    float *input = (float*)malloc((size_t)input_depth * INPUT_HT * INPUT_WD * sizeof(float));
    float* kernel = (float*)malloc((size_t)kernel_depth * KERNEL_HT * KERNEL_WD * sizeof(float));
    float* cpu_output = (float*)malloc((size_t)output_depth * OUTPUT_HT * OUTPUT_WD * sizeof(float));
    float* gpu_output = (float*)malloc((size_t)output_depth * OUTPUT_HT * OUTPUT_WD * sizeof(float));

    // Initialize input and kernel with some values
    for(int i = 0; i < (size_t)input_depth * INPUT_HT * INPUT_WD; i++)
    {
        input[i] = i % 10;
    }
    for (int i = 0; i < (size_t)kernel_depth * KERNEL_HT * KERNEL_WD; i++)
    {
        kernel[i] = i % 10;
    }
    //onv2d_cpu(input, kernel, cpu_output);
    //validate_outputs(cpu_output, 5); 
    //float *d_input, *d_kernel, *d_output;
    //cudaMalloc(&d_input, (size_t)inp_depth * INPUT_HT * INPUT_WD * sizeof(float));
    //cudaMalloc(&d_kernel, (size_t)kernel_depth * KERNEL_HT * KERNEL_WD * sizeof(float));
    //cudaMalloc(&d_output, (size_t)output_depth * OUTPUT_HT * OUTPUT_WD * sizeof(float));

    //cudaMemcpy(d_input, input, (size_t)inp_depth * INPUT_HT * INPUT_WD *sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_kernel, kernel, (size_t)kernel_depth * (size_t)C_in * KERNEL_HT * KERNEL_WD * sizeof(float), cudaMemcpyHostToDevice);

    //dim3 blockSize(16, 16, 1);
    //dim3 gridSize((OUTPUT_WD + blockSize.x - 1) / blockSize.x, 
    //              (OUTPUT_HT + blockSize.y - 1) / blockSize.y, C_out);

    // Launch the convolution kernel
    //conv2d<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, 
    //                                INPUT_HT, INPUT_WD, KERNEL_HT, KERNEL_WD, OUTPUT_HT, OUTPUT_WD,
    //                                C_in, C_out);
    //cudaMemcpy(gpu_output, d_output, (size_t)C_out * OUTPUT_HT * OUTPUT_WD * sizeof(float), cudaMemcpyDeviceToHost);
    //validate_outputs(cpu_output, gpu_output, 5); 

    conv3d_cpu(input, kernel, cpu_output);
    validate_outputs(cpu_output, 5);
    free(input);
    free(kernel);
    free(cpu_output);
    free(gpu_output);

    // cudaFree(d_input);
    // cudaFree(d_kernel);
    // cudaFree(d_output);
    return 0;
}

