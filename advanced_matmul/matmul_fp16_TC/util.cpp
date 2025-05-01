#include "util.h"
#include <omp.h>
#include <sys/time.h>
#include <cmath>
#include <cstdbool>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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


double get_time() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

/*
void check_matmul(half *A, half *B, float *C, int M, int N, int K) {
  printf("Validating...\n");


  // Allocate float matrices
  float *f32A = alloc_mat_float(M, K);
  float *f32B = alloc_mat_float(K, N);
  float *C_ans = alloc_mat_float(M, N);
  zero_mat_float(C_ans, M, N);

  for (int i = 0; i < M * K; ++i) {
    f32A[i] = (float)A[i];
  }

  for (int i = 0; i < K * N; ++i) {
    f32B[i] = (float)B[i];
  }

  //printf("1.**********\n");
  
#pragma omp parallel for collapse(2) num_threads(20)
//#pragma omp parallel for num_threads(20)
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        //C_ans[i * N + j] = C_ans[i * N + j] + (float)((A[i * K + k]) * (B[k * N + j]));
        C_ans[i * N + j] = C_ans[i * N + j] + f32A[i * K + k] * f32B[k * N + j];
        //float a = (float)A[i * K + k];
        //float b = (float)B[k * N + j];
        //C_ans[i * N + j] += a * b;
      }
    }
    //if (i % 50 == 0) printf("%d**********\n", i);
  }

  //printf("2.**********\n");
  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float c = C[i * N + j];
      float c_ans = C_ans[i * N + j];
      if ((fabsf(c) - fabs(c_ans)) > eps &&
          (c_ans == 0 || fabsf((fabs(c) - fabs(c_ans)) / c_ans) > eps)) {
        ++cnt;
        if (cnt <= thr)
          printf("C[%d][%d] : correct_value = %f, your_value = %f\n", i, j,
                 (float)c_ans, (float)c);
        if (cnt == thr + 1)
          printf("Too many error, only first %d values are printed.\n", thr);
        is_valid = false;
      }
    }
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}
*/

void check_matmul(half *A, half *B, float *C, int M, int N, int K) {
  printf("Validating...\n");

  float *C_ans = alloc_mat_float(M, N);
  zero_mat_float(C_ans, M, N);

#pragma omp parallel for num_threads(20)
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        C_ans[i * N + j] = C_ans[i * N + j] + (float)((A[i * K + k]) * (B[k * N + j]));
      }
    }
  }

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float c = C[i * N + j];
      float c_ans = C_ans[i * N + j];
      if ((fabsf(c) - fabs(c_ans)) > eps &&
          (c_ans == 0 || fabsf((fabs(c) - fabs(c_ans)) / c_ans) > eps)) {
        ++cnt;
        if (cnt <= thr)
          printf("C[%d][%d] : correct_value = %f, your_value = %f\n", i, j,
                 (float)c_ans, (float)c);
        if (cnt == thr + 1)
          printf("Too many error, only first %d values are printed.\n", thr);
        is_valid = false;
      }
    }
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}

void print_mat(half *m, int R, int C) {
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) { printf("%+.3f ", (float)(m[i * C + j])); }
    printf("\n");
  }
}

void print_mat_float(float *m, int R, int C) {
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) { printf("%+.3f ", (float)(m[i * C + j])); }
    printf("\n");
  }
}

half *alloc_mat(int R, int C) {
  half *m;
  CHECK_CUDA(cudaMallocHost(&m, sizeof(half) * R * C));
  return m;
}

float *alloc_mat_float(int R, int C) {
  float *m;
  CHECK_CUDA(cudaMallocHost(&m, sizeof(float) * R * C));
  return m;
}

void rand_mat(half *m, int R, int C) {
  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      m[i * C + j] = (half) ((float)rand() / RAND_MAX - 0.5);
    }
  }
}


void zero_mat(half *m, int R, int C) { memset(m, 0, sizeof(half) * R * C); }

void zero_mat_float(float *m, int R, int C) { memset(m, 0, sizeof(float) * R * C); }
