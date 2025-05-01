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

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;

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
  for (; k < K; k++) {
    //C[i * N + j] += A[i * K + k] * B[k * N + j];
    sum += A[i * K + k] * B[k * N + j];
  }
  C[i * N + j] = sum;
}

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(
      cudaMemcpy(A_gpu, _A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(B_gpu, _B, sizeof(float) * K * N, cudaMemcpyHostToDevice));

  //dim3 griddim((N + 16 - 1) / 16, (M + 16 - 1) / 16);
  dim3 griddim((N + 32 - 1) / 32, (M + 32 - 1) / 32);
  //dim3 blockdim(16, 16);
  dim3 blockdim(32, 32);
  matmul_kernel<<<griddim, blockdim>>>(A_gpu, B_gpu, C_gpu, M, N, K);

  CHECK_CUDA(
      cudaMemcpy(_C, C_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  CHECK_CUDA(cudaMalloc((void **) &A_gpu, sizeof(float) * M * K));
  CHECK_CUDA(cudaMalloc((void **) &B_gpu, sizeof(float) * K * N));
  CHECK_CUDA(cudaMalloc((void **) &C_gpu, sizeof(float) * M * N));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
