#pragma once

double get_time();

void check_matmul(float *A, float *B, float *C, int M, int N, int K);

void print_mat(float *m, int R, int C);

float* alloc_mat(int R, int C);

void rand_mat(float *m, int R, int C);

void zero_mat(float *m, int R, int C);
