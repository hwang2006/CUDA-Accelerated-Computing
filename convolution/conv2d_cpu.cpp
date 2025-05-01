#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Allocate a 4D tensor (N, C, H, W)
float* alloc_tensor(int N, int C, int H, int W) {
    return (float*) aligned_alloc(32, sizeof(float) * N * C * H * W);
}

// Initialize tensor with random values
void rand_tensor(float* m, int N, int C, int H, int W) {
    int L = N * C * H * W;
    for (int i = 0; i < L; i++) {
        m[i] = (float)rand() / RAND_MAX - 0.5f;
    }
}

// Simple conv2d forward kernel (no optimization)
void conv2d(float* out, float* in, float* filter,
            int N, int C, int H, int W, int K, int R, int S,
            int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
    int out_h = (H + 2 * pad_h - (R - 1) * dilation_h - 1) / stride_h + 1;
    int out_w = (W + 2 * pad_w - (S - 1) * dilation_w - 1) / stride_w + 1;

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    for (int c = 0; c < C; ++c) {
                        for (int r = 0; r < R; ++r) {
                            for (int s = 0; s < S; ++s) {
                                int ih = oh * stride_h - pad_h + r * dilation_h;
                                int iw = ow * stride_w - pad_w + s * dilation_w;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    //int in_idx = ((n * C + c) * H + ih) * W + iw;
                                    //int filt_idx = ((k * C + c) * R + r) * S + s;
                                    int in_idx = n * C * H * W + c * H * W + ih * W + iw;
                                    int filt_idx = k * C * R * S + c * R * S + r * S + s;
                                    sum += in[in_idx] * filter[filt_idx];
                                }
                            }
                        }
                    }
                    //out[((n * K + k) * out_h + oh) * out_w + ow] = sum;
                    out[n * K * out_h * out_w + k * out_h * out_w + oh * out_w + ow] = sum;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int N = 1, C = 3, H = 3, W = 3;
    int K = 3, R = 3, S = 3;
    int pad_h = 0, pad_w = 0;
    int stride_h = 1, stride_w = 1;
    int dilation_h = 1, dilation_w = 1;
    int print_tensor = 0;
    int num_iterations = 1;

    if (argc > 1 && strcmp(argv[1], "-h") == 0) {
        printf("Usage: ./conv2d [-ph] [-n num_iterations] N C H W K R S pad_h pad_w stride_h stride_w dilation_h dilation_w\n");
        printf("Options:\n");
        printf("     -p : print tensor. (default: off)\n");
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
        return 0;
    }

    int arg_idx = 1;
    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (strcmp(argv[arg_idx], "-p") == 0) print_tensor = 1;
        else if (strcmp(argv[arg_idx], "-n") == 0) {
            num_iterations = atoi(argv[++arg_idx]);
        }
        arg_idx++;
    }

    int* param_vars[] = {&N, &C, &H, &W, &K, &R, &S,
                     &pad_h, &pad_w, &stride_h, &stride_w,
                     &dilation_h, &dilation_w};
    int num_params = sizeof(param_vars) / sizeof(param_vars[0]);
 
    for (int i = 0; i < argc - arg_idx && i < num_params; ++i) {
        *param_vars[i] = atoi(argv[arg_idx + i]);
    }

    int out_h = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int out_w = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    printf("Problem size: N = %d, C = %d, H = %d, W = %d, K = %d, R = %d, S = %d\n", N, C, H, W, K, R, S);
    printf("              pad_h = %d, pad_w = %d, stride_h = %d, stride_w = %d\n", pad_h, pad_w, stride_h, stride_w);
    printf("              dilation_h = %d, dilation_w = %d\n", dilation_h, dilation_w);
    printf("Number of iterations: %d\n", num_iterations);
    printf("Print tensor: %s\n", print_tensor ? "on" : "off");

    float *input = alloc_tensor(N, C, H, W);
    float *filter = alloc_tensor(K, C, R, S);
    float *output = alloc_tensor(N, K, out_h, out_w);

    rand_tensor(input, N, C, H, W);
    rand_tensor(filter, K, C, R, S);

    clock_t start = clock();
    //#pragma omp parallel for 
    for (int i = 0; i < num_iterations; ++i) {
        conv2d(output, input, filter, N, C, H, W, K, R, S,
               pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    }
    clock_t end = clock();

    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Avg. time: %f sec\n", duration / num_iterations);

    free(input);
    free(filter);
    free(output);
    return 0;
}
