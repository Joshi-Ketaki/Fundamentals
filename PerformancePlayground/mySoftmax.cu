#include<stdio.h>
#include<math.h>

float find_max(float *input, int N)
{
    float max_val = input[0];
    for(int i = 1; i < N; i++)
    {
        if(input[i] > max_val)
            max_val = input[i];
    }
    return max_val;
}

void validate_outputs(float *cpu_output, float* gpu_output, int num_samples)
{
    printf("\nValidating first %d output elements...\n", num_samples);
    for (int s = 0; s < num_samples; s++)
    {
        int idx = rand() % num_samples;
        if(cpu_output[idx] != gpu_output[idx])
        {
            printf("\n Output mismatch!");
            return;
        }
        printf("\n Outputs match");
        printf("\n CPU Output[%d] = %.3f\n", idx,  cpu_output[idx]);
        printf("\n GPU Output[%d] = %.3f\n", idx, gpu_output[idx]);
    }
    printf("\n(If these values look finite and not NaN/Inf, indexing is likely correct.)\n");
}

void softmax_cpu(float *input, float *output, int N, int max_val)
{
    //Compute the exponentials and also the sum
    float sum = 0.0f;
    for(int i = 0; i < N; i++)
    {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Normalize the output
    for(int i = 0; i < N; i++)
    {
        output[i] = output[i] / sum;
    }
}

__global__ void softmax_kernel(float* d_input, float* d_output, int N)
{
    int idx = threadIdx.x;
    // Load inpt into shared memory
    extern __shared__ float sh_mem[];
    float *sh_input = sh_mem;
    float *max_val = sh_mem + N;
    float *sum = sh_mem + N + 1;

    if(idx < N)
        sh_input[idx] = d_input[idx];
    __syncthreads();

    // Compute the max value
    if (threadIdx.x == 0)
    {
        *max_val = sh_input[0];
        for(int i = 0; i < N; i++)
        {
            if(sh_input[i] > *max_val)
                *max_val = sh_input[i];
        }
    }
    __syncthreads();

    // Compute exponentials
    float e = expf(sh_input[idx] - *max_val);
    if(idx < N)
        sh_input[idx] = e;
    __syncthreads();

    //Compute sum
    if(threadIdx.x == 0)
    {
        *sum = 0.0f;
        for(int i = 0; i < N; i++)
        {
            *sum += sh_input[i];
        }
    }
    __syncthreads();

    //Normalize
    if(idx < N)
        d_output[idx] = sh_input[idx] / *sum;
    
}

int main()
{
    int N = 4;
    float max_val;;
    float *input = (float*) malloc(sizeof(float) * N);
    float *cpu_output = (float*) malloc(sizeof(float) * N);
    float *gpu_output = (float*) malloc(sizeof(float) * N);
    // Initialize input with some values
    for(int i = 0; i < N; i++)
    {
        input[i] = 1.0f * (rand() % 10);
    }

    // Find max value in input
    max_val = find_max(input, N);
    // Compute softmax on CPU
    softmax_cpu(input, cpu_output, N, max_val);

    // Compute softmax on GPU
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * N);

    cudaMemcpy(d_input, input, sizeof(float) * N, cudaMemcpyHostToDevice);
    softmax_kernel<<<1, N, (N + 2) * sizeof(float)>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_output, d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Validate outputs
    validate_outputs(cpu_output, gpu_output, N);

    free(input);
    free(cpu_output);
    free(gpu_output);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}