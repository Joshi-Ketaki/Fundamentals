// %%writefile naive_matmul.cu
// !ncc naive_matmul.cu -o naive_matmul -arch=sm_75
// !./naive_matmul 4 4 4 32
#include <iostream>
#include <cuda_fp16.h>
//#include <hip_fp16.h> 
#include <cuda_runtime.h>
#include <ctime>
#include <mma.h>
using namespace nvcuda;
using namespace std;

// globals
// Reference: https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
// https://www.pny.com/file%20library/company/support/product%20brochures/nvidia%20quadro/english/gv100-datasheet-highlights-web.pdf
double peakTflops_fp32 = 15.7e12; // 15.7 TFLOPS/s for GV100 single precision
double peakTflops_fp16 = 29.6e12; // 29.6 TFLOPS/s for GV100 half precision
double peakBw = 9e11; // 900 GB/s

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
// global counter for total flops
__device__ unsigned long long deviceFlops = 0ull, deviceBytes = 0ull, deviceHalfFlops = 0ull, deviceHalfBytes = 0ull;


void matInit(float* D, int rowDim, int colDim, float val) {
  for (int i = 0; i < rowDim * colDim; i++) 
	D[i] = val;
}

void matInit(half* D, int rowDim, int colDim, float val) {
  for (int i = 0; i < rowDim * colDim; i++)
        D[i] = __float2half(val++);
}

void dispMat(float* D, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) 
	    cout << D[i * N + j] << " ";
      cout << endl;
  }
}

void dispMat_half(const half* D, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      // convert to float for printing
      float v = __half2float(D[i * N + j]);
      cout << v << " ";
    }
    cout << endl;
  }
}

void convertToColMajor(half* hMat, half* hMatCol, int K, int N)
{
  for (int i = 0; i < K; i++)
    for (int j = 0; j < N; j++)
      hMatCol[j * K + i] = hMat[i * N + j];
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

__global__ void naive_matmul_fp16(int M, int N, int K, half* hA, half* hB, float* C, int block_size) {
  int row = blockIdx.y * block_size + threadIdx.y; //(threadIdx.x / block_size);  //threadIdx.y;
  int col = blockIdx.x * block_size + threadIdx.x; //(threadIdx.x % block_size); //threadIdx.x;
  unsigned long long localFlops = 0ull, localBytes = 0ull;
  if (row < M && col < N) {
    half sum = __float2half(0.0f);
    for (int i = 0; i < K; i++) {
      sum = __hadd(sum, __hmul(hA[row * K + i], hB[i * N + col])); // hA[row * K + i] * hB[i * N + col];
      localFlops += 2ull; // 1 multiply + 1 add
      localBytes += sizeof(half) * 2ull; // one for read of element in A and one for element in B
    }
    C[row * N + col] = __half2float(sum);
    localBytes += 1ull * sizeof(float); // one for element write of C
  }
  atomicAdd(&deviceHalfFlops, localFlops);
  atomicAdd(&deviceHalfBytes, localBytes);
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
  unsigned long long localFlops = 0ull;
  unsigned long long localBytes = 0ull;

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
	for (int i = 0; i < min(block_size, K-(t*block_size)); i++)
	{
		sum += As[threadIdx.y * block_size + i] * Bs[i * block_size + threadIdx.x];
		localFlops += 2ull;
	}
        __syncthreads();
	 
  }
  
  if(row < M && col < N) {
	C[row*N+col] = sum;
        localBytes += 1ull * sizeof(float); // one write to C
  }
  
  atomicAdd(&deviceFlops, localFlops);
  atomicAdd(&deviceBytes, localBytes);

}

__global__ void tiled_matmul_fp16(int M, int N, int K, half* A, half* B, float* C, int block_size) {

  // This is the outlook from one tile's point of view
  // calc row and col of output matrix
  int row = blockIdx.y * block_size + threadIdx.y;
  int col = blockIdx.x * block_size + threadIdx.x;

  // We are making tiles along K dimension
  // total number of tiles = ceil (K/tiles)
  // The below calc is to account for cases such as K%blk_size != 0 
  // && K > ceil (K/block_size)

  half sum = __float2half(0.0f);
  unsigned long long localFlops = 0ull;
  unsigned long long localBytes = 0ull;

  for(int t = 0; t < ((K+block_size-1)/block_size); t++)
  {
	  // Load tiles from global memory into shared memory
 	  extern __shared__ half shared_mem_half[];
	  half* As = shared_mem_half;
	  half* Bs = shared_mem_half + (block_size * block_size);
	
	  int Acol = t * block_size + threadIdx.x;
	  int Brow = t * block_size + threadIdx.y;

        // load them in row major format
	  As[threadIdx.y * block_size + threadIdx.x] =  (row < M && Acol < K) ? A[row * K +  Acol] : __float2half(0.0f);
	  Bs[threadIdx.y * block_size + threadIdx.x] = (Brow < M && col < N) ? B[Brow * N + col] : __float2half(0.0f);
    localBytes += 2ull * sizeof(half); // one load from A and one from B
	  __syncthreads();

	  // Multiply tiles
    // if tile size is not a multiple of 32, there cound be illegal accesses
    // so we calculate the max as wither if the entire tile or only how many elements will remain
    // which is K - t*TILE_SIZE. Note this will be 0 if K %32 = 0. else it shou;d give remianing lelemnts of the 
    // t tile.
	  for (int i = 0; i < min(block_size, K - (t*block_size)); i++)
	  {
		  sum = __hadd(sum, __hmul(As[threadIdx.y * block_size + i], Bs[i * block_size + threadIdx.x]));
		  localFlops += 2ull;
	  } 
     __syncthreads();
	 
  }
  
  if(row < M && col < N) {
	    C[row*N+col] = __half2float(sum);
      localBytes += 1ull * sizeof(float); // one write to C
  }
  
  atomicAdd(&deviceHalfFlops, localFlops);
  atomicAdd(&deviceHalfBytes, localBytes);

}

__global__ void tensor_core_matmul_fp16(half* dA, const half*  dB, float* C, int M, int N, int K)
{
    // Each warp computes one 16×16 tile of C
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;

    // Fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension in 16-wide tiles
    for (int k = 0; k < K; k += WMMA_K)
    {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;
        // Load the inputs
        wmma::load_matrix_sync(a_frag, &dA[aRow * K + aCol], K);      
        wmma::load_matrix_sync(b_frag, &dB[bRow * N + bCol], N);       
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
}


void printStats(unsigned long long flops, unsigned long long bytes, double ridgeIntensity)
{
  double arithIntensity = ((double)flops/bytes);
  cout << "Total number of operations ="<< flops << endl;
  cout << "Total number of bytes movement=" << bytes << endl;
  cout << "Arithematic Intensity =" << arithIntensity << " FLOPs/bytes" << endl;
  cout << "Ridge Intensity = " << ridgeIntensity << " FLOPs/byte" << endl;
  // double effThroughput = min(peakTflops_fp32, arithIntensity/peakBw);
  // cout << "Effective Throughput = " << effThroughput/1e12 << " TFLOPS/s" << endl;
  
  if(arithIntensity < ridgeIntensity)
        cout << "The kernel is MEMORY-BOUND" << endl;
  else
        cout << "The kernel is COMPUTE-BOUND" << endl;
}

void zeroOut()
{
  unsigned long long zero = 0ull;
  cudaMemcpyToSymbol(deviceFlops, &zero, sizeof(unsigned long long));
  cudaMemcpyToSymbol(deviceBytes, &zero, sizeof(unsigned long long));
  cudaMemcpyToSymbol(deviceHalfFlops, &zero, sizeof(unsigned long long));
  cudaMemcpyToSymbol(deviceHalfBytes, &zero, sizeof(unsigned long long));
}

int main(int argc, char *argv[]) {

  int M = 64; // atoi(argv[1]);
  int N = 64; // atoi(argv[2]);
  int K = 64; // atoi(argv[3]);
  int block_size = 32; // atoi(argv[4]);
  cudaError_t err;

  float *A = new float[M * K];
  half* hA = new half[M * K];
  float *B = new float[K * N];
  half* hB = new half[K*N];
  float *C = new float[M * N];
  
  matInit(A, M, K, 1);
  matInit(hA, M, K, 1.0f);
  matInit(B, K, N, 1);
  matInit(hB, K, N, 1.0f);
  matInit(C, M, N, 0); 

  float *dA, *dB, *dC;
  half *dhA, *dhB;
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


  double ridgeIntensity_fp32 = (double)peakTflops_fp32/peakBw;
  double ridgeIntensity_fp16 = (double)peakTflops_fp16/peakBw;

  //******************************** Naive Matmul *******************************
  unsigned long long hostFlops = 0ull, hostBytes = 0ull;
  /* zeroOut();
  naive_matmul<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, block_size); 
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) 
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  cout << "\n Naive Matmul" << endl;
  cudaMemcpyFromSymbol(&hostFlops, deviceFlops, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&hostBytes, deviceBytes, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
  printStats(hostFlops, hostBytes, ridgeIntensity_fp32);

  // ****************************** FP 16 Matmul ********************************* 
  hostFlops = 0ull; hostBytes = 0ull;
  zeroOut();
  naive_matmul_fp16<<<gridDim, blockDim>>>(M, N, K, (half*)dhA, (half*)dhB, dC, block_size);
  cout << "\n Fp16 Naive Matmul" << endl;
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) 
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&hostFlops, deviceHalfFlops, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&hostBytes, deviceHalfBytes, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
  printStats(hostFlops, hostBytes, ridgeIntensity_fp16);*/

  // ******************************* Tiled Matmul **********************************
  hostFlops = 0ull; hostBytes = 0ull;
  zeroOut();
  cout << "\n  Tiled Matmul" << endl;
  size_t sharedMemSize = 2 * block_size * block_size * sizeof(float); // for A and B tiles
  tiled_matmul<<<gridDim, blockDim, sharedMemSize>>>(M, N, K, dA, dB, dC, block_size);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) 
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&hostFlops, deviceFlops, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&hostBytes, deviceBytes, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
  printStats(hostFlops, hostBytes, ridgeIntensity_fp32);
 
  // ******************************* Tiled FP16 Matmul **********************************  
  hostFlops = 0ull; hostBytes = 0ull;
  zeroOut();
  cout << "\n  Tiled FP16 Matmul" << endl;
  //size_t sharedMemSize = 2 * block_size * block_size * sizeof(float); // for A and B tiles
  tiled_matmul_fp16<<<gridDim, blockDim, sharedMemSize>>>(M, N, K, (half*)dhA, (half*)dhB, dC, block_size);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) 
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&hostFlops, deviceHalfFlops, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
  cudaMemcpyFromSymbol(&hostBytes, deviceHalfBytes, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
  printStats(hostFlops, hostBytes, ridgeIntensity_fp16);

  /* ******************************** Tensor Core FP16 Matmul *******************************/
  /* dim3 threadsPerBlock(32, 1, 1); // one warp per block
  dim3 gridDim((N + WMMA_N - 1) / WMMA_N, 
                (M + WMMA_M - 1) / WMMA_M);
  tensor_core_matmul_fp16<<<gridDim, threadsPerBlock>>>((half*)dhA, (half*)dhB, dC, M, N, K);
  cudaDeviceSynchronize();
  cout << "\n  Tensor Core FP16 Matmul" << endl;
  err = cudaGetLastError();
  if (err != cudaSuccess) 
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);*/
  /* ******************************* Cleanup ************************************************/
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


/*
RESULTS: M=N=K= 64. BLOCK_SIZE=32
Ridge Intensity FP32 = 17.4444 FLOPs/byte
Ridge Intensity FP16 = 32.8889 FLOPs/byte
Total number of operations =524288

Naive Matmul
Total number of bytes movement=2113536        <------ Byte movement is high due to no reuse of data
Arithematic Intensity =0.248062 FLOPs/bytes   <------ AI is not so great. The kernel is memory bound

Naive Matmul FP16
Fp16 Naive Matmul
Total number of bytes movement=1064960        <------ Byte movement is a little better due to fp16.
Arithematic Intensity =0.492308 FLOPs/bytes   <------ AI has improved due to fp16. The kernel is still memory bound


Tiled Matmul
Total number of bytes movement=81920           <------ Byte movement has improved due to tiling.
Arithematic Intensity =6.4 FLOPs/bytes         <------ AI has improved due to tiling. The kernel is still memory bound though

Tiled Matmul FP16:
Total number of bytes movement=49152           <------ Byte movement has improved further due to fp16.
Arithematic Intensity =10.6667 FLOPs/bytes.    <------ AI has improved much more due to tiling and fp16. We are still memory bound though

*/