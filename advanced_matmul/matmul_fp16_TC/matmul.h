// matmul.h
#pragma once

#include <cuda_fp16.h>

// Tensor Core implementation
void matmul(half *_A, half *_B, float *_C, int M, int N, int K);
void matmul_init(int M, int N, int K);
void matmul_cleanup(half *_A, half *_B, float *_C, int M, int N, int K);

// Standard CUDA implementation
void matmul_standard(half *_A, half *_B, float *_C, int M, int N, int K);
void matmul_init_standard(int M, int N, int K);
void matmul_cleanup_standard(half *_A, half *_B, float *_C, int M, int N, int K);
