#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "convolution.cuh"
#include "util.h"

static bool print = false;
static bool validation = false;
static int N = 1;
static int C = 3;
static int H = 3;
static int W = 3;
static int K = 3;
static int R = 3;
static int S = 3;
static int pad_h = 0;
static int pad_w = 0;
static int stride_h = 1;
static int stride_w = 1;
static int dilation_h = 1;
static int dilation_w = 1;

static int num_iterations = 1;

static void print_help(const char *prog_name) {
  printf(
      "Usage: %s [-pvh] [-n num_iterations] N C H W K R S pad_h pad_w "
      "stride_h stride_w dilation_h dilation_w\n",
      prog_name);
  printf("Options:\n");
  printf("     -p : print tensor. (default: off)\n");
  printf("     -v : validate convolution. (default: off)\n");
  printf("     -h : print this page.\n");
  printf("     -n : number of iterations (default: 1)\n");
  printf("      N : batch size (default: 1)\n");
  printf("      C : input channel size (default: 3)\n");
  printf("      H : input height (default: 3)\n");
  printf("      W : input width (default: 3)\n");
  printf("      K : output channel size (default: 3)\n");
  printf("      R : filter height (default: 3)\n");
  printf("      S : filter width (default: 3)\n");
  printf("      pad_h : top and bottom padding (default: 0)\n");
  printf("      pad_w : left and right padding (default: 0)\n");
  printf("      stride_h : vertical stride (default: 1)\n");
  printf("      stride_w : horizontal stride (default: 1)\n");
  printf("      dilation_h : vertical dilation (default: 1)\n");
  printf("      dilation_w : horizontal dilation (default: 1)\n");
}

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:m:")) != -1) {
    switch (c) {
      case 'p': print = true; break;
      case 'v': validation = true; break;
      case 'n': num_iterations = atoi(optarg); break;
      case 'h':
      default: print_help(argv[0]); exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0: N = (size_t) atoi(argv[i]); break;
      case 1: C = (size_t) atoi(argv[i]); break;
      case 2: H = (size_t) atoi(argv[i]); break;
      case 3: W = (size_t) atoi(argv[i]); break;
      case 4: K = (size_t) atoi(argv[i]); break;
      case 5: R = (size_t) atoi(argv[i]); break;
      case 6: S = (size_t) atoi(argv[i]); break;
      case 7: pad_h = (size_t) atoi(argv[i]); break;
      case 8: pad_w = (size_t) atoi(argv[i]); break;
      case 9: stride_h = (size_t) atoi(argv[i]); break;
      case 10: stride_w = (size_t) atoi(argv[i]); break;
      case 11: dilation_h = (size_t) atoi(argv[i]); break;
      case 12: dilation_w = (size_t) atoi(argv[i]); break;
      default: break;
    }
  }

  printf(
      "Problem size: N = %d, C = %d, H = %d, W = %d, K = %d, R = %d, S = "
      "%d\n",
      N, C, H, W, K, R, S);
  printf("              pad_h = %d, pad_w = %d, stride_h = %d, stride_w = %d\n",
         pad_h, pad_w, stride_h, stride_w);
  printf("              dilation_h = %d, dilation_w = %d\n", dilation_h,
         dilation_w);
  printf("Number of iterations: %d\n", num_iterations);
  printf("Print tensor: %s\n", print ? "on" : "off");
  printf("Validation: %s\n", validation ? "on" : "off");
  printf("\n");
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);

  /* Allocate and initialize tensor on CPU */
  printf("Initializing... ");
  float *I, *F, *O, *BUF1, *BUF2;
  I = alloc_tensor(N, C, H, W);
  F = alloc_tensor(K, C, R, S);
  printf("done!\n");

  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  O = alloc_tensor(ON, OC, OH, OW);
  BUF1 = alloc_tensor(C, R, S, N * OH * OW);
  BUF2 = alloc_tensor(K, N, OH, OW);

  rand_tensor(I, N, C, H, W);
  rand_tensor(F, K, C, R, S);

  /* Initialize Convolution */
  convolution_initialize(N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w,
                         dilation_h, dilation_w);

  /* Run convolution for num_iterations */
  double elapsed_time_sum = 0;

  for (int i = 0; i < num_iterations; ++i) {
    printf("Calculating...(iter=%d) ", i);
    fflush(stdout);
    zero_tensor(O, ON, OC, OH, OW);
    double start_time = get_time();
    convolution(I, F, O, BUF1, BUF2, N, C, H, W, K, R, S, pad_h, pad_w,
                stride_h, stride_w, dilation_h, dilation_w);
    double elapsed_time = get_time() - start_time;
    printf("%f sec\n", elapsed_time);
    elapsed_time_sum += elapsed_time;
  }

  if (print) {
    printf("INPUT:\n");
    print_tensor(I, N, C, H, W);
    printf("FILTER:\n");
    print_tensor(F, K, C, R, S);
    printf("OUTPUT:\n");
    print_tensor(O, ON, OC, OH, OW);
  }

  /* Cleanup convolution */
  convolution_cleanup(I, F, O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                      stride_w, dilation_h, dilation_w);

  if (validation) {
    check_convolution(I, F, O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                      stride_w, dilation_h, dilation_w);
  }

  /* Print performance results */
  double elapsed_time_avg = elapsed_time_sum / num_iterations;
  printf("Avg. time: %f sec\n", elapsed_time_avg);
  printf("Avg. throughput: %f GFLOPS\n",
         2.0 * ON * OC * OH * OW * C * R * S / elapsed_time_avg / 1e9);

  return 0;
}
