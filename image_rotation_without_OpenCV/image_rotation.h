#pragma once

void rotate_image_naive(float *input_images, float *output_images, int W, int H,
                        float sin_theta, float cos_theta, int num_src_images);
void rotate_image(float *input_images, float *output_images, int W, int H,
                  float sin_theta, float cos_theta, int num_src_images);
void rotate_image_init(int W, int H, int num_src_images);
void rotate_image_cleanup();
