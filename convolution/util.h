#pragma once

double get_time();

float* alloc_tensor(int N, int C, int H, int W);

void rand_tensor(float *m, int N, int C, int H, int W);

void zero_tensor(float *m, int N, int C, int H, int W);

void print_tensor(float *m, int N, int C, int H, int W);

void check_convolution(float *I, float *F, float *O, int N, int C, int H, int W,
                       int K, int R, int S, int pad_h, int pad_w, int stride_h,
                       int stride_w, int dilation_h, int dilation_w);