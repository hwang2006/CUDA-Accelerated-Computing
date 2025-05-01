#pragma once

void convolution(float *_I, float *_F, float *_O, float *_BUF1, float *_BUF2,
                 int N, int C, int H, int W, int K, int R, int S, int pad_h,
                 int pad_w, int stride_h, int stride_w, int dilation_h,
                 int dilation_w);

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w);

void convolution_cleanup(float *_I, float *_F, float *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w);