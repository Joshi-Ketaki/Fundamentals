// %%writefile naive_matmul.cu
// !ncc naive_matmul.cu -o naive_matmul -arch=sm_75
// !./naive_matmul 4 4 4 32
#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ctime>
using namespace std;

// global counter for total flops
__device__ unsigned long long deviceFlops = 0ull, deviceBytes = 0ull, deviceHalfFlops = 0ull, deviceHalfBytes = 0ull;

// globals
double peakPerf = 15.7e12; // 15.7 TFLOPS/s peak perf for GV100 single precision
double peakBw = 9e11; // 900 GB/s

void matInit(float* D, int rowDim, int colDim, int val) {
  for (int i = 0; i < rowDim * colDim; i++) 
	D[i] = val;
}

void matInit(half* D, int rowDim, int colDim, int val) {
  for (int i = 0; i < rowDim * colDim; i++)
        D[i] = val;
}

void dispMat(float* D, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) 
	cout << D[i * N + j] << " ";
    cout << endl;
  }
}

__global__ void naive_matmul(int M, int N, int K, float* A, float* B, float* C, int block_size) {
  int row = blockIdx.y * block_size + threadIdx.y; //(threadIdx.x / block_size);  //threadIdx.y;
  int col = blockIdx.x * block_size + threadIdx.x; //(threadIdx.x % block_size); //threadIdx.x;
  unsigned long long localFlops = 0ull, localBytes = 0ull;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
      sum += A[row * K + i] * B[i * N + col];
      localFlops += 2ull; // 1 multiply + 1 add
      localBytes += sizeof(float) * 2ull; // one for read of element in A and one for element in B
    }
    C[row * N + col] = sum;  
    localBytes += 1ull * sizeof(float); // one for element write of C
    
  atomicAdd(&deviceFlops, localFlops);
  atomicAdd(&deviceBytes, localBytes);
}
}

__global__ void naive_matmul_fp16(int M, int N, int K, half* hA, half* hB, float* fC, int block_size) {
  int row = blockIdx.y * block_size + threadIdx.y; //(threadIdx.x / block_size);  //threadIdx.y;
  int col = blockIdx.x * block_size + threadIdx.x; //(threadIdx.x % block_size); //threadIdx.x;
  int localFlops = 0, localBytes = 0;
  if (row < M && col < N) {
    half sum = 0.0f;
    for (int i = 0; i < K; i++) {
      sum += __hadd(sum, __hmul(hA[row * K + i], hB[i * N + col])); // hA[row * K + i] * hB[i * N + col];
      localFlops += 2; // 1 multiply + 1 add
      localBytes += sizeof(half) * 2; // one for read of element in A and one for element in B
    }
    fC[row * N + col] = __half2float(sum);
    localBytes += sizeof(float); // one for element write of C
  }
  atomicAdd(&deviceHalfFlops, localFlops);
  atomicAdd(&deviceHalfBytes, localBytes);
}

double printStats(unsigned long long flops, unsigned long long bytes, double ridgeIntensity)
{
  double arithIntensity = ((double)flops/bytes);
  cout << "Total number of operations ="<< flops << endl;
  cout << "Total number of bytes movement=" << bytes << endl;
  cout << "Arithematic Intensity =" << arithIntensity << " FLOPs/bytes" << endl;
  cout << "Ridge Intensity = " << ridgeIntensity << " FLOPs/byte" << endl;
  double effThroughput = min(peakPerf, arithIntensity/peakBw);
  
  if(arithIntensity < ridgeIntensity)
        cout << "The kernel is MEMORY-BOUND" << endl;
  else
        cout << "The kernel is COMPUTE-BOUND" << endl;
}

int main(int argc, char *argv[]) {

  int M = 64; // atoi(argv[1]);
  int N = 64; // atoi(argv[2]);
  int K = 64; // atoi(argv[3]);
  int block_size = 32; // atoi(argv[4]);

  float *A = new float[M * K];
  half* hA = new half[M * K];
  float *B = new float[K * N];
  half* hB = new half[K*N];
  float *C = new float[M * N];
  
  matInit(A, M, K, 1);
  matInit(hA, M, K, __float2half(1));
  matInit(B, K, N, 1);
  matInit(hB, K, N, __float2half(1));
  matInit(C, M, N, 0); 

  float *dA, *dhA, *dB, *dhB, *dC;
  cudaMalloc(&dA, sizeof(float) * M * K);
  cudaMalloc(&dhA, sizeof(half) * M * K);
  cudaMalloc(&dB, sizeof(float) * K * N);
  cudaMalloc(&dhB, sizeof(half) * K * N);
  cudaMalloc(&dC, sizeof(float) * M * N);

  cudaMemcpy(dA, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dhA, hA, sizeof(half) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dhB, hB, sizeof(half) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  dim3 blockDim(block_size, block_size, 1);
  dim3 gridDim((N + block_size - 1) / block_size,
             (M + block_size - 1) / block_size);


  double ridgeIntensity = (double)peakPerf/peakBw;

  //******************************** Naive Matmul *******************************
  unsigned long long hostFlops = 0, hostBytes = 0; 
  unsigned long long zero = 0;
  cudaMemcpyToSymbol(deviceFlops, &zero, sizeof(unsigned long long));
  cudaMemcpyToSymbol(deviceBytes, &zero, sizeof(unsigned long long));
  naive_matmul<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, block_size);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  cout << "\n Naive Matmul" << endl;
  cudaMemcpyFromSymbol(&hostFlops, deviceFlops, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&hostBytes, deviceBytes, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
  printStats(hostFlops, hostBytes, ridgeIntensity);

  // ****************************** FP 16 Matmul ********************************* 
  //hostFlops = 0ull;
  //hostBytes = 0ull;
  /*cout << "\n Fp16 Naive Matmul" << endl;
  naive_matmul_fp16<<<gridDim, blockDim>>>(M, N, K, dhA, dhB, dC, block_size);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&hostFlops, deviceHalfFlops, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&hostBytes, deviceHalfBytes, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
  printStats(hostFlops, hostBytes, ridgeIntensity);
  dispMat(C, M, N);*/

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] hA;
  delete[] hB;

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dhA);
  cudaFree(dhB);

  return 0;
}
