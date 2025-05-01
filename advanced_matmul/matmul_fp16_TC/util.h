#pragma once
#include <cuda_fp16.h>

double get_time();

void check_matmul(half *A, half *B, float *C, int M, int N, int K);

void print_mat(half *m, int R, int C);

void print_mat_float(float *m, int R, int C);

half* alloc_mat(int R, int C);

float* alloc_mat_float(int R, int C);

void rand_mat(half *m, int R, int C);

void zero_mat(half *m, int R, int C);

void zero_mat_float(float *m, int R, int C);
