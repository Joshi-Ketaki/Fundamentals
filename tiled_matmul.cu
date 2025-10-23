// %%writefile naive_matmul.cu
// !ncc naive_matmul.cu -o naive_matmul -arch=sm_75
// !./naive_matmul 4 4 4 32
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__device__ unsigned long long deviceFlops, deviceBytes;

void matInit(float* D, int rowDim, int colDim, int val) {
  for (int i = 0; i < rowDim * colDim; i++) D[i] = val;
}

void dispMat(float* D, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) cout << D[i * N + j] << " ";
    cout << endl;
  }
}

__global__ void tiled_matmul(int M, int N, int K, float* A, float* B, float* C, int block_size) {

  // This is the outlook from one tile's point of view
  // calc row and col of output matrix
  int row = blockIdx.y * block_size + threadIdx.y;
  int col = blockIdx.x * block_size + threadIdx.x;

  // We are making tiles along K dimension
  // total number of tiles = ceil (K/tiles)
  // The below calc is to account for cases such as K%blk_size != 0 
  // && K > ceil (K/block_size)

  float sum = 0.0f;
  unsigned long long localFlops = 0;
  unsigned long long localBytes = 0;

  for(int t = 0; t < ((K+block_size-1)/block_size); t++)
  {
	// Load tiles from global memory into shared memory
 	extern __shared__ float shared_mem[];
	float* As = shared_mem;
	float* Bs = shared_mem + (block_size * block_size);
	
	int Acol = t * block_size + threadIdx.x;
	int Brow = t * block_size + threadIdx.y;

        // load them in row major format
	As[threadIdx.y * block_size + threadIdx.x] =  (row < M && Acol < N) ? A[row * K +  Acol] : 0.0f;
	Bs[threadIdx.y * block_size + threadIdx.x] = (Brow < M && col < N) ? B[Brow * N + col] : 0.0f;
        localBytes += 2 * sizeof(float); // one load from A and one from B
	__syncthreads();

	// Multiply tiles
	for (int i = 0; i < block_size; i++)
	{
		sum += As[threadIdx.y * block_size + i] * Bs[i * block_size + threadIdx.x];
		localFlops += 2;
	}
        __syncthreads();
	 
  }
  
  if(row < M && col < N) {
	C[row*N+col] = sum;
        localBytes += sizeof(float); // one write to C
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
  
  double peakPerf = 14 * 10^12; // 14 TFLOPS/s peak perf for GV100 single precision
  double peakBw = 900; // 900 GB/s 

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
  // Using 2D here.Makes it easy to visualize compute in the kernel
  dim3 blockDim(block_size, block_size, 1);
  // dim3 gridDim(grid_x, grid_y, 1);
  dim3 gridDim((N + block_size - 1) / block_size,
             (M + block_size - 1) / block_size);


  float sharedMemSize = 2 * (block_size * block_size) * sizeof(float);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  tiled_matmul<<<gridDim, blockDim, sharedMemSize>>>(M, N, K, dA, dB, dC, block_size);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float kernelExecutionTime;
  cudaEventElapsedTime(&kernelExecutionTime, start, stop);
  

  cudaDeviceSynchronize();

  unsigned long long hostFlops = 0, hostBytes = 0;
  cudaMemcpyFromSymbol(&hostFlops, deviceFlops, sizeof(unsigned long long));
  cudaMemcpyFromSymbol(&hostBytes, deviceBytes, sizeof(unsigned long long));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("Kernel launch error: %s\n", cudaGetErrorString(err));

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("Kernel execution error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  //dispMat(C, M, N);


  cout << "**** Roofline Analysis ****"<< endl;
  cout << "Total number of operations = "<< hostFlops << endl;
  cout << "Total number of bytes movement = " << hostBytes << endl;
  // cout << "Arithematic Intensity = " << ((double)hostFlops/hostBytes) << " FLOPs/bytes" << endl;
  // cout << "Ridge Intensity = " << (peakPerf/peakBw);
  // cout << "Kernel Execution Time = " << kernelExecutionTime << " milliseconds" << endl;
  // cout << "Performance = " << (double)hostFlops/(kernelExecutionTime * 0.001) << " FLOPs/seconds" << endl;

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
