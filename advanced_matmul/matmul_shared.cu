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

#define BLOCK_SIZE 32

static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                                     int K) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int gj = blockIdx.x, gi = blockIdx.y;
  int lj = threadIdx.x, li = threadIdx.y;
  if (gi * BLOCK_SIZE >= M || gj * BLOCK_SIZE >= N) return;

  __shared__ float Alocal[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Blocal[BLOCK_SIZE][BLOCK_SIZE];
  float c = 0.f;

  for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
    int Ai = gi * BLOCK_SIZE + li, Bj = gj * BLOCK_SIZE + lj;
    int Aj = bk + lj, Bi = bk + li;
    Alocal[li][lj] = (Ai < M && Aj < K) ? A[Ai * K + Aj] : 0.f;
    Blocal[li][lj] = (Bi < K && Bj < N) ? B[Bi * N + Bj] : 0.f;
    __syncthreads();

    for (int lk = 0; lk < BLOCK_SIZE; ++lk) {
      c += Alocal[li][lk] * Blocal[lk][lj];
    }
    __syncthreads();
  }

  if (i < M && j < N) C[i * N + j] = c;
}

static float *A_gpu, *B_gpu, *C_gpu;

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(
      cudaMemcpy(A_gpu, _A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(B_gpu, _B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  matmul_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(
      cudaMemcpy(_C, C_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost));
}

void matmul_init(int M, int N, int K) {
  CHECK_CUDA(cudaMalloc(&A_gpu, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_gpu, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_gpu, M * N * sizeof(float)));
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));
}