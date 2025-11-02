#include<stdio.h>
#include<math.h>

float find_max(float *input, int N, float max_val)
{
    max_val = input[0];
    for(int i = 1; i < N; i++)
    {
        if(input[i] > max_val)
            max_val = input[i];
    }
    return max_val;
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


int main()
{
    int N = 4;
    float max_val;
    float *input = (float*) malloc(sizeof(float) * N);
    float *output = (float*) malloc(sizeof(float) * N);
    // Initialize input with some values
    for(int i = 0; i < N; i++)
    {
        input[i] = 1.0f * (rand() % 10);
    }

    // Find max value in input
    max_val = find_max(input, N, max_val);
    // Compute softmax on CPU
    softmax_cpu(input, output, N, max_val);

    for(int i = 0; i < N; i++)
    {
        printf("Softmax Output[%d] = %.6f\n", i, output[i]);
    }
    return 0;
}