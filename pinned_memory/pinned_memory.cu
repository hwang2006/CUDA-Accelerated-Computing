#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

float measure_bandwidth_async(const char* label, void* dst, const void* src,
                size_t bytes, cudaMemcpyKind kind, int repetitions, cudaStream_t stream) {
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // Warm-up
  CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, kind, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < repetitions; ++i) {
    CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, kind, stream));
  }
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  float avg_time = ms / repetitions;
  float bandwidth = (bytes / avg_time / 1e6);  // GB/s

  printf("%-30s: %7.2f GB/s (%zu bytes, %d reps)\n", label, bandwidth, bytes, repetitions);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return bandwidth;
}

int main() {
  std::vector<size_t> sizes = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1<< 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28};// 1MB, 4MB, 16MB, 64MB, 256MB
  const int repetitions = 10;

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  for (size_t bytes : sizes) {
    printf("\n===== Buffer size: %zu bytes (%.2f MB) =====\n", bytes, bytes / (1024.0 * 1024.0));

    // Allocate device memory
    int *d_mem;
    CHECK_CUDA(cudaMalloc(&d_mem, bytes));

    // 1. Pageable memory
    {
      int *h_pageable = (int*) malloc(bytes);
      if (!h_pageable) {
        fprintf(stderr, "Failed to allocate pageable memory\n");
        return EXIT_FAILURE;
      }

      measure_bandwidth_async("H2D Pageable (async)", d_mem, h_pageable, bytes, cudaMemcpyHostToDevice, repetitions, stream);
      measure_bandwidth_async("D2H Pageable (async)", h_pageable, d_mem, bytes, cudaMemcpyDeviceToHost, repetitions, stream);

      free(h_pageable);
    }

    // 2. Pinned memory
    {
      int *h_pinned;
      CHECK_CUDA(cudaMallocHost(&h_pinned, bytes));

      measure_bandwidth_async("H2D Pinned (async)", d_mem, h_pinned, bytes, cudaMemcpyHostToDevice, repetitions, stream);
      measure_bandwidth_async("D2H Pinned (async)", h_pinned, d_mem, bytes, cudaMemcpyDeviceToHost, repetitions, stream);

      CHECK_CUDA(cudaFreeHost(h_pinned));
    }

    CHECK_CUDA(cudaFree(d_mem));
  }

  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
