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

void check_convolution(half *I, half *F, float *O, int N, int C, int H, int W,
                       int K, int R, int S, int pad_h, int pad_w, int stride_h,
                       int stride_w, int dilation_h, int dilation_w) {
  float *O_ans;
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  O_ans = alloc_tensor_float(ON, OC, OH, OW);
  zero_tensor_float(O_ans, ON, OC, OH, OW);

#pragma omp parallel for
  for (int on = 0; on < ON; ++on) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float sum = 0;
          for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
              for (int s = 0; s < S; ++s) {
                const int n = on;
                const int h = oh * stride_h - pad_h + r * dilation_h;
                const int w = ow * stride_w - pad_w + s * dilation_w;
                const int k = oc;
                if (h < 0 || h >= H || w < 0 || w >= W) continue;
                sum += (float)(I[(((size_t)n * C + c) * H + h) * W + w] *
                       F[(((size_t)k * C + c) * R + r) * S + s]);
              }
            }
          }
          O_ans[(((size_t)on * OC + oc) * OH + oh) * OW + ow] = sum;
        }
      }
    }
  }

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int on = 0; on < ON; ++on) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float o = O[(((size_t)on * OC + oc) * OH + oh) * OW + ow];
          float o_ans = O_ans[(((size_t)on * OC + oc) * OH + oh) * OW + ow];
          if (fabsf(fabsf(o) - fabsf(o_ans)) > eps &&
              (o_ans == 0 || fabsf((fabsf(o) - fabsf(o_ans)) / fabsf(o_ans)) > eps)) {
            ++cnt;
            if (cnt <= thr)
              printf(
                  "O[%d][%d][%d][%d] : correct_value = %f, your_value = %f\n",
                  on, oc, oh, ow, (float)o_ans, (float)o);
            if (cnt == thr + 1)
              printf("Too many error, only first %d values are printed.\n",
                     thr);
            is_valid = false;
          }
        }
      }
    }
  }

  if (is_valid) {
    printf("Validation Result: VALID\n");
  } else {
    printf("Validation Result: INVALID\n");
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

void rand_mat_float(float *m, int R, int C) {
  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      m[i * C + j] = ((float)rand() / RAND_MAX - 0.5);
    }
  }
}

half* alloc_tensor( int N, int C, int H, int W) {
  half *m;
  CHECK_CUDA(cudaMallocHost(&m, (size_t)N * C * H * W * sizeof(half)));
  return m;
}

float* alloc_tensor_float( int N, int C, int H, int W) {
  float *m;
  CHECK_CUDA(cudaMallocHost(&m, (size_t)N * C * H * W * sizeof(float)));
  return m;
}


void rand_tensor(half *m, int N, int C, int H, int W) {
  size_t L = N * C * H * W;
  for (size_t j = 0; j < L; j++) { m[j] = (half) ((float)rand() / RAND_MAX - 0.5);; }
}

void rand_tensor_float(float *m, int N, int C, int H, int W) {
  size_t L = N * C * H * W;
  for (size_t j = 0; j < L; j++) { m[j] = ((float) rand() / RAND_MAX - 0.5); }
}


void zero_tensor(half *m, int N, int C, int H, int W) {
  size_t L = (size_t)N * C * H * W;
  memset(m, 0, sizeof(half) * L);
}

void zero_tensor_float(float *m, int N, int C, int H, int W) {
  size_t L = (size_t)N * C * H * W;
  memset(m, 0, sizeof(float) * L);
}


void print_tensor(half *m, int N, int C, int H, int W) {
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      printf("Batch %d, Channel %d\n", n, c);
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          printf("%+.3f ", (float) (m[(((size_t)n * C + c) * H + h) * W + w]));
        }
        printf("\n");
      }
    }
  }
}

void print_tensor_float(float *m, int N, int C, int H, int W) {
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      printf("Batch %d, Channel %d\n", n, c);
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          printf("%+.3f ", m[(((size_t)n * C + c) * H + h) * W + w]);
        }
        printf("\n");
      }
    }
  }
}


void zero_mat(half *m, int R, int C) { memset(m, 0, sizeof(half) * R * C); }

void zero_mat_float(float *m, int R, int C) { memset(m, 0, sizeof(float) * R * C); }

