#include <cstdio>

#include "matmul.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define LOOP_I(_loop) for(int i=0; i < _loop; i++)

#define COL_SIZE (32)
#define ROW_SIZE (32)

//#define COL_SIZE (64)
//#define ROW_SIZE (16)

//#define BLOCKS 8
#define BLOCKS 16

static int Mbegin[BLOCKS], Mend[BLOCKS];
static cudaStream_t upload_stream, download_stream, calc_stream;
static cudaEvent_t upload_events[BLOCKS], calc_events[BLOCKS];

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;

void naive_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        _C[i * N + j] += _A[i * K + k] * _B[k * N + j];
      }
    }
  }
}
/*
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {

   int ix = threadIdx.x + blockIdx.x * blockDim.x; //column
   int iy = threadIdx.y + blockIdx.y * blockDim.y; //row
 
   if (iy >= M || ix >= N) return;
   float sum = 0.0f;
   for (int k = 0; k < K; k++) sum += A[iy * K + k] * B[k * N + ix];
   C[iy * N + ix] = sum;
}
*/
/*
// matmul_kernel_xRow
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                                     int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x; //col
  int j = blockDim.y * blockIdx.y + threadIdx.y; //row
  if (i >= N || j >= M) return;
  float sum = 0.0;
  for (int k = 0; k < K; ++k) sum += A[j * K + k] * B[k * N + i];
  C[j * N + i] = sum;
}
*/

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    // Calculate the row and column for this thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within valid range
    if (col >= N || row >= M) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;

}


__global__ void matmul_kernel_xRow(float *A, float *B, float *C, int M, int N, int K) {

   int row = threadIdx.x + blockIdx.x * blockDim.x; 
   int col = threadIdx.y + blockIdx.y * blockDim.y; 

   //if ( col >= M || row >= N) return;
   if ( col >= N || row >= M) return;
   float sum = 0.0f;
   for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
   C[row * N + col] = sum;
}


/*
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    // Calculate the row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within valid range
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
*/

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Remove this line after you complete the matmul on GPU
  //naive_cpu_matmul(_A, _B, _C, M, N, K);

  // (TODO) Upload A and B matrix to GPU
  //CHECK_CUDA(cudaMemcpy(A_gpu, _A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  //CHECK_CUDA(cudaMemcpy(B_gpu, _B, K * N * sizeof(float), cudaMemcpyHostToDevice));  

  // upload stream
  CHECK_CUDA(cudaMemcpyAsync(B_gpu, _B, K * N * sizeof(float),
                             cudaMemcpyHostToDevice, upload_stream));
  LOOP_I(BLOCKS)
  { 
    CHECK_CUDA(cudaMemcpyAsync(&A_gpu[Mbegin[i] * K], &_A[Mbegin[i] * K],
                               (Mend[i] - Mbegin[i]) * K * sizeof(float),
                               cudaMemcpyHostToDevice, upload_stream));
    CHECK_CUDA(cudaEventRecord(upload_events[i], upload_stream));
  }

  // calc stream
  LOOP_I(BLOCKS)
  {
    //dim3 blockDim(16, 16);
    //dim3 gridDim((N + 16 - 1) / 16, (Mend[i] - Mbegin[i] + 16 - 1) / 16);


    dim3 blockDim(COL_SIZE, ROW_SIZE);
    //dim3 gridDim((N + COL_SIZE - 1)/COL_SIZE, (M + ROW_SIZE -1)/ROW_SIZE);
    dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (M + blockDim.y -1)/blockDim.y);

    CHECK_CUDA(cudaStreamWaitEvent(calc_stream, upload_events[i]));
    matmul_kernel<<<gridDim, blockDim, 0, calc_stream>>>(
        &A_gpu[Mbegin[i] * K], B_gpu, &C_gpu[Mbegin[i] * N],
        (Mend[i] - Mbegin[i]), N, K);
    CHECK_CUDA(cudaEventRecord(calc_events[i], calc_stream));

  }

  // download stream
  LOOP_I(BLOCKS)
  {
    CHECK_CUDA(cudaStreamWaitEvent(download_stream, calc_events[i]));
    CHECK_CUDA(cudaMemcpyAsync(&_C[Mbegin[i] * N], &C_gpu[Mbegin[i] * N],
                               (Mend[i] - Mbegin[i]) * N * sizeof(float),
                               cudaMemcpyDeviceToHost, download_stream)); 
  }
  // (TODO) Launch kernel on a GPU
  //dim3 blockDim(COL_SIZE, ROW_SIZE);
  //dim3 gridDim((N + COL_SIZE - 1)/COL_SIZE, (M + ROW_SIZE -1)/ROW_SIZE);
  //dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (M + blockDim.y -1)/blockDim.y);

  //dim3 blockDim(32, 32);
  //dim3 gridDim((M + 32 - 1) / 32, (N + 32 - 1) / 32);
  //fprintf(stdout, "Grid dimensions: %d x %d blocks of %d x %d threads\n",
  //         			gridDim.x, gridDim.y, blockDim.x, blockDim.y);
  //fflush(stdout);
  //matmul_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);


  //(TODO) Launch kernel with xRow on a GPU
  //dim3 gridDim((M + COL_SIZE - 1)/COL_SIZE, (N + ROW_SIZE -1)/ROW_SIZE);
  //dim3 gridDim((M + blockDim.x - 1)/blockDim.x, (N + blockDim.y -1)/blockDim.y);
  //matmul_kernel_xRow<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);
  //CHECK_CUDA(cudaGetLastError());  

  // (TODO) Download C matrix from GPU
  //CHECK_CUDA(cudaMemcpy(_C, C_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost));  

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  // (TODO) Allocate device memory
  CHECK_CUDA(cudaMalloc(&A_gpu, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_gpu, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_gpu, M * N * sizeof(float)));
  
  LOOP_I(BLOCKS)
  {
    Mbegin[i] = M / BLOCKS * i;
    Mend[i] = M / BLOCKS * (i + 1);
    if (i == BLOCKS - 1) Mend[i] = M;
  }

  //Create streams
  CHECK_CUDA(cudaStreamCreate(&upload_stream));
  CHECK_CUDA(cudaStreamCreate(&download_stream));
  CHECK_CUDA(cudaStreamCreate(&calc_stream));

  //Create events
  LOOP_I(BLOCKS)
  {
   CHECK_CUDA(cudaEventCreate(&upload_events[i]));
   CHECK_CUDA(cudaEventCreate(&calc_events[i]));
  }
  
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  // (TODO) Do any post-matmul cleanup work here.
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));
  CHECK_CUDA(cudaStreamDestroy(upload_stream));
  CHECK_CUDA(cudaStreamDestroy(download_stream));
  CHECK_CUDA(cudaStreamDestroy(calc_stream));
  LOOP_I(BLOCKS)
  {
    CHECK_CUDA(cudaEventDestroy(upload_events[i]));
    CHECK_CUDA(cudaEventDestroy(calc_events[i]));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
