// %%writefile naive_matmul.cu
// !ncc naive_matmul.cu -o naive_matmul -arch=sm_75
// !./naive_matmul 4 4 4 32
#include <iostream>
#include <cuda_runtime.h>
#include <ctime>
using namespace std;

// global counter for total flops
__device__ unsigned long long deviceFlops = 0, deviceBytes = 0;

void matInit(float* D, int rowDim, int colDim, int val) {
  for (int i = 0; i < rowDim * colDim; i++) D[i] = val;
}

void dispMat(float* D, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) cout << D[i * N + j] << " ";
    cout << endl;
  }
}

__global__ void naive_matmul(int M, int N, int K, float* A, float* B, float* C, int block_size) {
  int row = blockIdx.y * block_size + threadIdx.y; //(threadIdx.x / block_size);  //threadIdx.y;
  int col = blockIdx.x * block_size + threadIdx.x; //(threadIdx.x % block_size); //threadIdx.x;
  unsigned long long localFlops = 0, localBytes = 0;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
      sum += A[row * K + i] * B[i * N + col];
      localFlops += 2; // 1 multiply + 1 add
      localBytes += sizeof(float) * 2; // one for read of element in A and one for element in B
    }
    C[row * N + col] = sum;  
    localBytes += sizeof(float); // one for element write of C
  }
  atomicAdd(&deviceFlops, localFlops);
  atomicAdd(&deviceBytes, localBytes);
}

int main(int argc, char *argv[]) {
  /*if (argc != 5) {
    cout << "Usage: " << argv[0] << " <M> <N> <K> <block_size>" << endl;
    return 1;
  }*/

  int M = 64; // atoi(argv[1]);
  int N = 64; // atoi(argv[2]);
  int K = 64; // atoi(argv[3]);
  int block_size = 32; // atoi(argv[4]);

  float *A = new float[M * K];
  float *B = new float[K * N];
  float *C = new float[M * N];

  double peakPerf = 15.7e12; // 14 TFLOPS/s peak perf for GV100 single precision
  double peakBw = 9e11; // 900 GB/s
  
  matInit(A, M, K, 1);
  matInit(B, K, N, 1);
  matInit(C, M, N, 0); // initialize to 5 to detect if overwritten

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

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start); // Record when kernel starts
  naive_matmul<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, block_size);
  cudaEventRecord(stop);  // Record when kernel stops

  cudaEventSynchronize(stop); // Wait for kernel to finish

  float kernelExecutionTime;
  cudaEventElapsedTime(&kernelExecutionTime, start, stop); // time recorded in milliseconds
 
  unsigned long long hostFlops = 0, hostBytes = 0;
  cudaMemcpyFromSymbol(&hostFlops, deviceFlops, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&hostBytes, deviceBytes, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));

  
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("Kernel execution error: %s\n", cudaGetErrorString(err));
 
  cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  // dispMat(C, M, N);

  cout << "**** Occupancy Calculations ****" << endl;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
  cout << "Threads per block = " << threadsPerBlock << endl;
  cout << "Device max threads per SM: " << prop.maxThreadsPerMultiProcessor << endl;
  cout << "Device max threads per block: " << prop.maxThreadsPerBlock << endl;
  cout << "Device SM count: " << prop.multiProcessorCount << endl;
  // calc theoritical occupancy
  int maxActiveBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM,
                                                naive_matmul,
                                                threadsPerBlock,
                                                0);


  // cudaDeviceProp prop;
  // cudaGetDeviceProperties(&prop, 0);

  cout << "\n (Theoretical) maxActiveBlocksPerSM = " << maxActiveBlocksPerSM << endl;
  double occupancy = (double)(maxActiveBlocksPerSM * threadsPerBlock) /
                   (double)prop.maxThreadsPerMultiProcessor * 100.00f;
  cout << "\n Theoretical Occupancy = " << occupancy << "%" << endl;


  double ridgeIntensity = (peakPerf/peakBw);
  double arithIntensity = ((double)hostFlops/hostBytes);
  cout << "\n**** Roofline Analysis ****"<< endl;
  cout << "Total number of operations ="<< hostFlops << endl;
  cout << "Total number of bytes movement=" << hostBytes << endl;
  cout << "Arithematic Intensity =" << arithIntensity << " FLOPs/bytes" << endl;
  cout << "Ridge Intensity = " << ridgeIntensity << " FLOPs/byte" << endl;
  cout << "Kernel Execution Time=" << kernelExecutionTime << " milliseconds" << endl;
  cout << "Achieved performance =" << (double)hostFlops/(kernelExecutionTime * 0.001) << " FLOPs/seconds" << endl;


  if(arithIntensity < ridgeIntensity)
	cout << "The kernel is MEMORY-BOUND" << endl;
  else
 	cout << "The kernel is COMPUTE-BOUND" << endl;
 
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  delete[] A;
  delete[] B;
  delete[] C;
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
