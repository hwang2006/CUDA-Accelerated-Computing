#pragma once

void matmul(float *_A, float *_B, float *_C, int M, int N, int K);
void matmul_init(int M, int N, int K);
void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K);
