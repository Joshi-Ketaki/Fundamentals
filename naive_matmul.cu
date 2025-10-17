// %%writefile naive_matmul.cu
// !ncc naive_matmul.cu -o naive_matmul -arch=sm_75
// !./naive_matmul 4 4 4 32
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

void matInit(float* D, int rowDim, int colDim, int val) {
  for (int i = 0; i < rowDim * colDim; i++) D[i] = val;
}

void dispMat(float* D, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) cout << D[i * N + j] << " ";
    cout << endl;
  }
}

__global__ void naive_matmul(int M, int N, int K, float* A, float* B, float* C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
      sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
  }
}

int main(int argc, char *argv[]) {
  /*if (argc != 5) {
    cout << "Usage: " << argv[0] << " <M> <N> <K> <block_size>" << endl;
    return 1;
  }*/

  int M = 8; // atoi(argv[1]);
  int N = 8; // atoi(argv[2]);
  int K = 8; // atoi(argv[3]);
  int block_size = 4; // atoi(argv[4]);

  float *A = new float[M * K];
  float *B = new float[K * N];
  float *C = new float[M * N];
  matInit(A, M, K, 1);
  matInit(B, K, N, 2);
  matInit(C, M, N, 5); // initialize to 5 to detect if overwritten

  float *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof(float) * M * K);
  cudaMalloc(&dB, sizeof(float) * K * N);
  cudaMalloc(&dC, sizeof(float) * M * N);

  cudaMemcpy(dA, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  // int grid_x = (N + block_size - 1) / block_size;
  // int grid_y = (M + block_size - 1) / block_size;
  dim3 blockDim(block_size, block_size, 1);
  // dim3 gridDim(grid_x, grid_y, 1);
  dim3 gridDim((N + block_size - 1) / block_size,
             (M + block_size - 1) / block_size);


  naive_matmul<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("Kernel execution error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  dispMat(C, M, N);

  delete[] A;
  delete[] B;
  delete[] C;
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
