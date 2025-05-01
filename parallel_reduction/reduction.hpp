#pragma once
#include <cstddef>  // for size_t

// Constants for kernel configuration
//#define THREADS_PER_BLOCK 1024
#define THREADS_PER_BLOCK 512
#define ELEMENTS_PER_BLOCK (THREADS_PER_BLOCK * 2)

// Host function declaration
double reduction_gpu(double* input_host, size_t num_elements);

#ifdef __CUDACC__  // Only compile kernel code when CUDA is involved
__global__ void reduce_partial_sum(double* input, double* partial_sum, size_t num_elements);
#endif
