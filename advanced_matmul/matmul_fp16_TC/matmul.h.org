#pragma once

#include <cuda_fp16.h>

void matmul(half *_A, half *_B, float *_C, int M, int N, int K);
void matmul_init(int M, int N, int K);
void matmul_cleanup(half *_A, half *_B, float *_C, int M, int N, int K);
