#include "util.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdbool>
#include <cmath>
#include <sys/time.h>
#include <omp.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void check_vecadd(float *A, float *B, float *C, int N) {
  printf("Validating...\n");

  float *C_ans = alloc_vec(N);
  zero_vec(C_ans, N);

#pragma omp parallel for num_threads(20)
  for (int i = 0; i < N; ++i) {
    C_ans[i] = A[i] + B[i];
  }

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < N; ++i) {
    float c = C[i];
    float c_ans = C_ans[i];
    if (fabsf(c - c_ans) > eps && (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("C[%d] : correct_value = %f, your_value = %f\n", i, c_ans, c);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
      is_valid = false;
    }
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}

void print_vec(float *m, int N) {
  for (int i = 0; i < N; ++i) { 
    printf("%+.3f ", m[i]);
  }
  printf("\n");
}

float* alloc_vec(int N) {
  float *m = (float *) aligned_alloc(32, sizeof(float) * N);
  return m;
}

void rand_vec(float *m, int N) {
  for (int i = 0; i < N; i++) { 
    m[i] = (float) rand() / RAND_MAX - 0.5;
  }
}

void zero_vec(float *m, int N) {
  memset(m, 0, sizeof(float) * N);
}
