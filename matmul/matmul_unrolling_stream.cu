#include <cstdio>

#include "matmul.h"

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                              int K) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (j >= N || i >= M) return;

  // init C
  //C[i * N + j] = 0;
  float sum = 0.0f;

  float a0, a1, a2, a3, a4, a5, a6, a7;
  float b0, b1, b2, b3, b4, b5, b6, b7;
  int k;

  // loop unrolling
  for (k = 0; k + 7 < K; k += 8) {
    a0 = A[i * K + (k + 0)];
    a1 = A[i * K + (k + 1)];
    a2 = A[i * K + (k + 2)];
    a3 = A[i * K + (k + 3)];
    a4 = A[i * K + (k + 4)];
    a5 = A[i * K + (k + 5)];
    a6 = A[i * K + (k + 6)];
    a7 = A[i * K + (k + 7)];
    b0 = B[(k + 0) * N + j];
    b1 = B[(k + 1) * N + j];
    b2 = B[(k + 2) * N + j];
    b3 = B[(k + 3) * N + j];
    b4 = B[(k + 4) * N + j];
    b5 = B[(k + 5) * N + j];
    b6 = B[(k + 6) * N + j];
    b7 = B[(k + 7) * N + j];
    //C[i * N + j] += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 +
    sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 +
                    a6 * b6 + a7 * b7;
  }

  // Deal with trailing k
  //for (; k < K; k++) { C[i * N + j] += A[i * K + k] * B[k * N + j]; }
  for (; k < K; k++) { sum += A[i * K + k] * B[k * N + j]; }
  C[i * N + j] = sum;
}

#define BLOCKS 8

static int Mbegin[BLOCKS], Mend[BLOCKS];
static cudaStream_t upload_stream, download_stream, calc_stream;
static cudaEvent_t upload_events[BLOCKS], calc_events[BLOCKS];
static float *A_gpu, *B_gpu, *C_gpu;

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(cudaMemcpyAsync(B_gpu, _B, K * N * sizeof(float),
                             cudaMemcpyHostToDevice, upload_stream));
  for (int i = 0; i < BLOCKS; i++) {
    CHECK_CUDA(cudaMemcpyAsync(&A_gpu[Mbegin[i] * K], &_A[Mbegin[i] * K],
                               (Mend[i] - Mbegin[i]) * K * sizeof(float),
                               cudaMemcpyHostToDevice, upload_stream));
    CHECK_CUDA(cudaEventRecord(upload_events[i], upload_stream));
  }

  for (int i = 0; i < BLOCKS; i++) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 16 - 1) / 16, (Mend[i] - Mbegin[i] + 16 - 1) / 16);

    CHECK_CUDA(cudaStreamWaitEvent(calc_stream, upload_events[i]));
    matmul_kernel<<<gridDim, blockDim, 0, calc_stream>>>(
        &A_gpu[Mbegin[i] * K], B_gpu, &C_gpu[Mbegin[i] * N],
        (Mend[i] - Mbegin[i]), N, K);
    CHECK_CUDA(cudaEventRecord(calc_events[i], calc_stream));
  }

  for (int i = 0; i < BLOCKS; i++) {
    CHECK_CUDA(cudaStreamWaitEvent(download_stream, calc_events[i]));
    CHECK_CUDA(cudaMemcpyAsync(&_C[Mbegin[i] * N], &C_gpu[Mbegin[i] * N],
                               (Mend[i] - Mbegin[i]) * N * sizeof(float),
                               cudaMemcpyDeviceToHost, download_stream));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  for (int i = 0; i < BLOCKS; i++) {
    Mbegin[i] = M / BLOCKS * i;
    Mend[i] = M / BLOCKS * (i + 1);
    if (i == BLOCKS - 1) Mend[i] = M;
  }

  CHECK_CUDA(cudaStreamCreate(&upload_stream));
  CHECK_CUDA(cudaStreamCreate(&download_stream));
  CHECK_CUDA(cudaStreamCreate(&calc_stream));
  for (int i = 0; i < BLOCKS; i++) {
    CHECK_CUDA(cudaEventCreate(&upload_events[i]));
    CHECK_CUDA(cudaEventCreate(&calc_events[i]));
  }

  CHECK_CUDA(cudaMalloc(&A_gpu, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_gpu, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_gpu, M * N * sizeof(float)));
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));
  CHECK_CUDA(cudaStreamDestroy(upload_stream));
  CHECK_CUDA(cudaStreamDestroy(download_stream));
  CHECK_CUDA(cudaStreamDestroy(calc_stream));
  for (int i = 0; i < BLOCKS; i++) {
    CHECK_CUDA(cudaEventDestroy(upload_events[i]));
    CHECK_CUDA(cudaEventDestroy(calc_events[i]));
  }
}
