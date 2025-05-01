#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iomanip>  // for std::setw

// Utility function to print a 2D matrix (row-major)
void print_matrix(const float* data, int rows, int cols, int col_width = 6) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(col_width) << std::setprecision(3) << data[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

int compute_output_dim(int input, int pad, int dilation, int kernel, int stride) {
    return (input + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

void conv2d_naive(const float* input, const float* weights, float* output,
                  int N, int C, int H, int W,
                  int K, int R, int S,
                  int pad, int stride, int dilation) {
    int OH = compute_output_dim(H, pad, dilation, R, stride);
    int OW = compute_output_dim(W, pad, dilation, S, stride);

    std::fill(output, output + N * K * OH * OW, 0.0f);

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < OH; ++h) {
                for (int w = 0; w < OW; ++w) {
                    float acc = 0.0f;
                    for (int c = 0; c < C; ++c) {
                        for (int r = 0; r < R; ++r) {
                            for (int s = 0; s < S; ++s) {
                                int in_h = -pad + h * stride + r * dilation;
                                int in_w = -pad + w * stride + s * dilation;
                                if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                                    acc += input[n * C * H * W + c * H * W + in_h * W + in_w] *
                                           weights[k * C * R * S + c * R * S + r * S + s];
                                }
                            }
                        }
                    }
                    output[n * K * OH * OW + k * OH * OW + h * OW + w] = acc;
                }
            }
        }
    }
}

void im2col(const float* input, int C, int H, int W,
            int R, int S, int pad, int stride, int dilation,
            float* col, int OH, int OW) {
    int channels_col = C * R * S;

    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int col_row = (c * R + r) * S + s;
                for (int h = 0; h < OH; ++h) {
                    for (int w = 0; w < OW; ++w) {
                        int im_row = -pad + h * stride + r * dilation;
                        int im_col = -pad + w * stride + s * dilation;
                        int col_col = h * OW + w;
                        if (im_row >= 0 && im_row < H && im_col >= 0 && im_col < W) {
                            col[col_row * OH * OW + col_col] =
                                input[c * H * W + im_row * W + im_col];
                        } else {
                            col[col_row * OH * OW + col_col] = 0;
                        }
                    }
                }
            }
        }
    }
}

void conv2d_im2col(int N, int C, int H, int W,
                   int K, int R, int S,
                   int pad, int stride, int dilation,
                   const std::vector<float>& input,
                   const std::vector<float>& weights,
                   std::vector<float>& output) {
    int OH = compute_output_dim(H, pad, dilation, R, stride);
    int OW = compute_output_dim(W, pad, dilation, S, stride);
    int col_size = C * R * S * OH * OW;
    std::vector<float> col(col_size);

    for (int n = 0; n < N; ++n) {
        im2col(input.data() + n * C * H * W, C, H, W, R, S, pad, stride, dilation,
               col.data(), OH, OW);

        std::cout << "\n--- Weight matrix reshaped (K=" << K << ", C*R*S=" << C * R * S << ") ---\n";
        print_matrix(weights.data(), K, C * R * S, 2);

        std::cout << "\n--- im2col transformed matrix (C*R*S x OH*OW) ---\n";
        print_matrix(col.data(), C * R * S, OH * OW, 2);

        for (int k = 0; k < K; ++k) {
            for (int col_col = 0; col_col < OH * OW; ++col_col) {
                float acc = 0.0f;
                for (int col_row = 0; col_row < C * R * S; ++col_row) {
                    acc += weights[k * C * R * S + col_row] *
                           col[col_row * OH * OW + col_col];
                }
                output[n * K * OH * OW + k * OH * OW + col_col] = acc;
            }
        }
    }
}

int main(int argc, char** argv) {
    int N = 1, C = 1, H = 5, W = 5;
    int K = 1, R = 3, S = 3;
    int pad = 1, stride = 1, dilation = 1;

    if (argc == 11) {
        N = std::atoi(argv[1]); C = std::atoi(argv[2]); H = std::atoi(argv[3]); W = std::atoi(argv[4]);
        K = std::atoi(argv[5]); R = std::atoi(argv[6]); S = std::atoi(argv[7]);
        pad = std::atoi(argv[8]); stride = std::atoi(argv[9]); dilation = std::atoi(argv[10]);
    } else {
        std::cout << "Usage: ./conv2d_im2col N C H W K R S pad stride dilation\nUsing default values.\n";
    }

    std::cout << "\nParsed Parameters:\n";
    std::cout << "N=" << N << " C=" << C << " H=" << H << " W=" << W << "\n";
    std::cout << "K=" << K << " R=" << R << " S=" << S << "\n";
    std::cout << "pad=" << pad << " stride=" << stride << " dilation=" << dilation << "\n";

    int OH = compute_output_dim(H, pad, dilation, R, stride);
    int OW = compute_output_dim(W, pad, dilation, S, stride);
    int input_size = N * C * H * W;
    int weight_size = K * C * R * S;
    int output_size = N * K * OH * OW;

    std::vector<float> input(input_size, 1.0f);
    std::vector<float> weights(weight_size, 3.0f);
    std::vector<float> output_im2col(output_size);
    std::vector<float> output_naive(output_size);

    conv2d_im2col(N, C, H, W, K, R, S, pad, stride, dilation, input, weights, output_im2col);
    conv2d_naive(input.data(), weights.data(), output_naive.data(),
                 N, C, H, W, K, R, S, pad, stride, dilation);

    bool correct = true;
    for (int i = 0; i < output_size; ++i) {
        if (std::fabs(output_naive[i] - output_im2col[i]) > 1e-4f) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": naive=" << output_naive[i]
                      << ", im2col=" << output_im2col[i] << "\n";
            break;
        }
    }

    std::cout << "\nOutput (N=" << N << ", K=" << K << ", OH=" << OH << ", OW=" << OW << "):\n";
    for (int n = 0; n < N; n++){
     std::cout << "* Batch number: " << n << "\n";
      for (int k = 0; k < K; ++k){
       std::cout << "** Output channel(" << k << ")\n";
        for (int i = 0; i < OH; ++i) {
         for (int j = 0; j < OW; ++j) {
            std::cout << output_im2col[i * OW + j] << " ";
        }
        std::cout << "\n";
       }
      std::cout << "\n";
     }
    std::cout << "\n";
   }

    std::cout << (correct ? "** Outputs match!\n" : "** Outputs differ!\n");
    return 0;
}
