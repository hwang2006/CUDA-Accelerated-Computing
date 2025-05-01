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

//static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                                     int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= M || j >= N) return;
  float sum = 0.0;
  for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[k * N + j];
  C[i * N + j] = sum;
}

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

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(
      cudaMemcpy(A_gpu, _A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(B_gpu, _B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 blockDim(32, 32);
  dim3 gridDim((M + 32 - 1) / 32, (N + 32 - 1) / 32);
  matmul_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(
      cudaMemcpy(_C, C_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  CHECK_CUDA(cudaMalloc(&A_gpu, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_gpu, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_gpu, M * N * sizeof(float)));

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
