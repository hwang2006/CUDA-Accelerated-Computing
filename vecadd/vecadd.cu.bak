#include <cstdio>

#include "vecadd.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void vecadd_kernel(const int N, const float *a, const float *b, float *c) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  c[tidx] = a[tidx] + b[tidx];
}

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;

void vecadd(float *_A, float *_B, float *_C, int N) {
  // (TODO) Upload A and B vector to GPU

  // Launch kernel on a GPU
  dim3 gridDim(N / 512);
  dim3 blockDim(512);
  vecadd_kernel<<<gridDim, blockDim>>>(N, A_gpu, B_gpu, C_gpu);

  // (TODO) Download C vector from GPU

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void vecadd_init(int N) {
  // (TODO) Allocate device memory

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void vecadd_cleanup(float *_A, float *_B, float *_C, int N) {
  // (TODO) Do any post-vecadd cleanup work here.

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
