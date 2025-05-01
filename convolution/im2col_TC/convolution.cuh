#pragma once

#include <cuda_fp16.h>
#include <cstdio>

void convolution(half *_I, half *_F, float *_O, half *_BUF1, float *_BUF2,
                 int N, int C, int H, int W, int K, int R, int S, int pad_h,
                 int pad_w, int stride_h, int stride_w, int dilation_h,
                 int dilation_w);

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w);

void convolution_cleanup(half *_I, half *_F, float *_O, half* BUF1_, float* BUF2_, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w);