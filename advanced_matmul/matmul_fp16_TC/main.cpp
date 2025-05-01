#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_fp16.h>
#include "util.h"
#include "matmul.h"

static void print_help(const char* prog_name) {
    printf("Usage: %s [-pvh] [-n num_iterations] [-m mode] M N K\n", prog_name);
    printf("Options:\n");
    printf("  -p : print matrix data. (default: off)\n");
    printf("  -v : validate matrix multiplication. (default: off)\n");
    printf("  -h : print this page.\n");
    printf("  -n : number of iterations (default: 1)\n");
    printf("  -m : execution mode ('tensor_core' or 'standard') (default: tensor_core)\n");
    printf("    M : number of rows of matrix A and C. (default: 8)\n");
    printf("    N : number of columns of matrix B and C. (default: 8)\n");
    printf("    K : number of columns of matrix A and rows of B. (default: 8)\n");
}

static bool print_matrix = false;
static bool validation = false;
static int M = 8, N = 8, K = 8;
static int num_iterations = 1;
static char mode[20] = "tensor_core";

static void parse_opt(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "pvhn:m:")) != -1) {
        switch (c) {
            case 'p':
                print_matrix = true;
                break;
            case 'v':
                validation = true;
                break;
            case 'n':
                num_iterations = atoi(optarg);
                break;
            case 'm':
                strncpy(mode, optarg, sizeof(mode) - 1);
                mode[sizeof(mode) - 1] = '\0'; // Ensure null termination
                break;
            case 'h':
            default:
                print_help(argv[0]);
                exit(0);
        }
    }
    for (int i = optind, j = 0; i < argc; ++i, ++j) {
        switch (j) {
            case 0: M = atoi(argv[i]); break;
            case 1: N = atoi(argv[i]); break;
            case 2: K = atoi(argv[i]); break;
            default: break;
        }
    }
    printf("Options:\n");
    printf("  Problem size: M = %d, N = %d, K = %d\n", M, N, K);
    printf("  Number of iterations: %d\n", num_iterations);
    printf("  Execution mode: %s\n", mode);
    printf("  Print matrix: %s\n", print_matrix ? "on" : "off");
    printf("  Validation: %s\n", validation ? "on" : "off");
    printf("\n");
}

int main(int argc, char **argv) {
    parse_opt(argc, argv);

    printf("Initializing (%s mode)... ", mode); fflush(stdout);
    half *A = alloc_mat(M, K);
    half *B = alloc_mat(K, N);
    float *C = alloc_mat_float(M, N);
    rand_mat(A, M, K);
    rand_mat(B, K, N);

    if (strcmp(mode, "tensor_core") == 0) {
        matmul_init(M, N, K);
    } else if (strcmp(mode, "standard") == 0) {
        matmul_init_standard(M, N, K);
    } else {
        fprintf(stderr, "Error: Invalid execution mode '%s'. Use 'tensor_core' or 'standard'.\n", mode);
        exit(EXIT_FAILURE);
    }
    printf("done!\n"); fflush(stdout);

    double elapsed_time_sum = 0;
    for (int i = 0; i < num_iterations; ++i) {
        printf("Calculating (%s mode)...(iter=%d) ", mode, i); fflush(stdout);
        zero_mat_float(C, M, N);
        double start_time = get_time();
        if (strcmp(mode, "tensor_core") == 0) {
            matmul(A, B, C, M, N, K);
        } else if (strcmp(mode, "standard") == 0) {
            matmul_standard(A, B, C, M, N, K);
        }
        double elapsed_time = get_time() - start_time;
        printf("%f sec\n", elapsed_time);
        elapsed_time_sum += elapsed_time;
    }

    if (print_matrix) {
        printf("MATRIX A:\n"); print_mat(A, M, K);
        printf("MATRIX B:\n"); print_mat(B, K, N);
        printf("MATRIX C:\n"); print_mat_float(C, M, N);
    }

    if (strcmp(mode, "tensor_core") == 0) {
        matmul_cleanup(A, B, C, M, N, K);
    } else if (strcmp(mode, "standard") == 0) {
        matmul_cleanup_standard(A, B, C, M, N, K);
    }

    if (validation) {
        check_matmul(A, B, C, M, N, K);
    }

    double elapsed_time_avg = elapsed_time_sum / num_iterations;
    printf("Avg. time (%s mode): %f sec\n", mode, elapsed_time_avg);
    printf("Avg. throughput (%s mode): %f GFLOPS\n", mode, 2.0 * M * N * K / elapsed_time_avg / 1e9);

    return 0;
}
