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

half* alloc_tensor( int N, int C, int H, int W);

float* alloc_tensor_float( int N, int C, int H, int W);

void rand_tensor(half *m, int N, int C, int H, int W);

void rand_tensor_float(float *m, int N, int C, int H, int W);

void zero_tensor(half *m, int N, int C, int H, int W);

void zero_tensor_float(float *m, int N, int C, int H, int W);

void check_convolution(half *I, half *F, float *O, int N, int C, int H, int W,
                       int K, int R, int S, int pad_h, int pad_w, int stride_h,
                       int stride_w, int dilation_h, int dilation_w);

void print_tensor(half *m, int N, int C, int H, int W);
void print_tensor_float(float *m, int N, int C, int H, int W);
