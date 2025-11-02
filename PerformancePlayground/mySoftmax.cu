#include<stdio.h>
#include<math.h>

void softmax_cpu(float *input, float *output, int N)
{
    float max_val = -INFINITY;
    // Find the maximum value for numerical stability
    for(int i = 0; i < N; i++)
    {
        if(input[i] > max_val)
            max_val = input[i];
    }

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
    float *input = (float*) malloc(sizeof(float) * N);
    float *output = (float*) malloc(sizeof(float) * N);
    // Initialize input with some values
    for(int i = 0; i < N; i++)
    {
        input[i] = 1.0f * (rand() % 10);
    }

    // Compute softmax on CPU
    softmax_cpu(input, output, N);

    for(int i = 0; i < N; i++)
    {
        printf("Softmax Output[%d] = %.6f\n", i, output[i]);
    }
    return 0;
}