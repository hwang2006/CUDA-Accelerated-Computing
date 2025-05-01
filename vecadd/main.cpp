#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>

#include "util.h"
#include "vecadd.h"

static void print_help(const char* prog_name) {
  printf("Usage: %s [-pvh]  [-n num_iterations] N\n", prog_name);
  printf("Options:\n");
  printf("  -p : print vector data. (default: off)\n");
  printf("  -v : validate vector addition. (default: off)\n");
  printf("  -h : print this page.\n");
  printf("  -t : number of threads (default: 1)\n");
  printf("  -n : number of iterations (default: 1)\n");
  printf("   N : number of elements of vectors. (default: 8)\n");
}

static bool print_vector = false;
static bool validation = false;
static int N = 8;
static int num_iterations = 1;

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:")) != -1) {
    switch (c) {
      case 'p':
        print_vector = true;
        break;
      case 'v':
        validation = true;
        break;
      case 'n':
        num_iterations = atoi(optarg);
        break;
      case 'h':
      default:
        print_help(argv[0]);
        exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0: N = atoi(argv[i]); break;
      default: break;
    }
  }
  printf("Options:\n");
  printf("  Problem size: N = %d\n", N);
  printf("  Number of iterations: %d\n", num_iterations);
  printf("  Print vector: %s\n", print_vector ? "on" : "off");
  printf("  Validation: %s\n", validation ? "on" : "off");
  printf("\n");
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);

  printf("Initializing... "); fflush(stdout);
  float *A = alloc_vec(N);
  float *B = alloc_vec(N);
  float *C = alloc_vec(N);
  rand_vec(A, N);
  rand_vec(B, N);
  vecadd_init(N);
  printf("done!\n");

  double elapsed_time_sum = 0;
  for (int i = 0; i < num_iterations; ++i) {
    printf("Calculating...(iter=%d) ", i); fflush(stdout);
    zero_vec(C, N);
    double start_time = get_time();
    vecadd(A, B, C, N);
    double elapsed_time = get_time() - start_time;
    printf("%f sec\n", elapsed_time);
    elapsed_time_sum += elapsed_time;
  }

  if (print_vector) {
    printf("Vector A:\n"); print_vec(A, N);
    printf("Vector B:\n"); print_vec(B, N);
    printf("Vector C:\n"); print_vec(C, N);
  }

  vecadd_cleanup(A, B, C, N);

  if (validation) {
    check_vecadd(A, B, C, N);
  }

  double elapsed_time_avg = elapsed_time_sum / num_iterations;
  printf("Avg. time: %f sec\n", elapsed_time_avg);
  printf("Avg. throughput: %f GFLOPS\n", N / elapsed_time_avg / 1e9);

  return 0;
}
