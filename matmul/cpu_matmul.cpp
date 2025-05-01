#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cmath>

void init_matrix(std::vector<float>& mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

bool compare_matrices(const std::vector<float>& a, const std::vector<float>& b, int rows, int cols, float epsilon = 1e-3f) {
    int t = 0;
    for (int i = 0; i < rows * cols; ++i) {
        if (std::fabs(a[i] - b[i]) > epsilon) {
           if ( t < 5) { //print out the first 5 mismatche items
              printf("  **Mismatch: a[%d][%d] : %f, b[%d] : %f \n", i/cols, i%cols, a[i], i, b[i]);
              t++; continue;
           }
           return false;
        }
    }
    return true;
}

void naive_cpu_matmul(const std::vector<float>& A, const std::vector<float>& B,
                      std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void ikj_cpu_matmul(const std::vector<float>& A, const std::vector<float>& B,
                    std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void scalar_sum_cpu_matmul(const std::vector<float>& A, const std::vector<float>& B,
                           std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

/*
 - Why does tile_cpu_matmul() produce mismatches in some elements with large matrics like 2048 * 2048?
  Tiled (32): 54.876764 sec (match: NO)
  **Mismatch: a[0][1281] : 507.197021, b[1281] : 507.198120
  **Mismatch: a[0][1984] : 511.328094, b[1984] : 511.327057
  **Mismatch: a[1][1172] : 494.952881, b[3220] : 494.951752
  **Mismatch: a[2][303] : 519.900330, b[4399] : 519.899231
  **Mismatch: a[2][508] : 511.450562, b[4604] : 511.449554
  Tiled (64): 54.757527 sec (match: NO)
  **Mismatch: a[0][1281] : 507.197021, b[1281] : 507.198212
  **Mismatch: a[0][1464] : 527.072754, b[1464] : 527.071716
  **Mismatch: a[0][1997] : 512.214600, b[1997] : 512.213562
  **Mismatch: a[1][1172] : 494.952881, b[3220] : 494.951813
  **Mismatch: a[2][303] : 519.900330, b[4399] : 519.899292
 */
void tiled_cpu_matmul(const std::vector<float>& A, const std::vector<float>& B,
                      std::vector<float>& C, int M, int N, int K, int TILE_SIZE) {
    for (int ii = 0; ii < M; ii += TILE_SIZE) {
        for (int jj = 0; jj < N; jj += TILE_SIZE) {
            for (int kk = 0; kk < K; kk += TILE_SIZE) {
                for (int i = ii; i < std::min(ii + TILE_SIZE, M); ++i) {
                    for (int j = jj; j < std::min(jj + TILE_SIZE, N); ++j) {
                        float sum = 0.0f;
                        for (int k = kk; k < std::min(kk + TILE_SIZE, K); ++k)
                            sum += A[i * K + k] * B[k * N + j];
                        C[i * N + j] += sum;  
                        //double sum = 0.0; // Changed to double
                        //for (int k = kk; k < std::min(kk + TILE_SIZE, K); ++k)
                        //    sum += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
                        //C[i * N + j] += static_cast<float>(sum); // Cast back to float when storing
                    }
                }
            }
        }
    }
}


void tiled_cpu_matmul_correct(const std::vector<float>& A, const std::vector<float>& B,
                               std::vector<float>& C, int M, int N, int K, int TILE_SIZE) {
    for (int ii = 0; ii < M; ii += TILE_SIZE) {
        for (int jj = 0; jj < N; jj += TILE_SIZE) {
            for (int kk = 0; kk < K; kk += TILE_SIZE) {
                for (int i = ii; i < std::min(ii + TILE_SIZE, M); ++i) {
                    for (int j = jj; j < std::min(jj + TILE_SIZE, N); ++j) {
                        for (int k = kk; k < std::min(kk + TILE_SIZE, K); ++k) {
                            C[i * N + j] += A[i * K + k] * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

template <typename Func>
double run_and_time(int n_runs, int M, int N, int K,
                    Func matmul_func,
                    const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    using clock = std::chrono::high_resolution_clock;
    double total_time = 0.0;
    for (int i = 0; i < n_runs; ++i) {
        std::fill(C.begin(), C.end(), 0.0f);
        auto start = clock::now();
        matmul_func(A, B, C, M, N, K);
        auto end = clock::now();
        total_time += std::chrono::duration<double>(end - start).count();
    }
    return total_time / n_runs;
}

int main() {
    std::vector<int> sizes = {64, 128, 256, 400, 512, 1024, 1200, 2048};
    std::vector<int> tile_sizes = {32, 64};
    const int n_runs = 1;

    for (int size : sizes) {
        int M = size, N = size, K = size;
        std::vector<float> A(M * K), B(K * N);
        std::vector<float> C1(M * N), C2(M * N), C3(M * N), C4(M * N);

        init_matrix(A, M, K);
        init_matrix(B, K, N);

        //Ensure C Matrics are zeroed
        std::fill(C1.begin(), C1.end(), 0.0f); 
        std::fill(C2.begin(), C2.end(), 0.0f); 
        std::fill(C3.begin(), C3.end(), 0.0f); 
        std::fill(C4.begin(), C4.end(), 0.0f);

        double t1 = run_and_time(n_runs, M, N, K, naive_cpu_matmul, A, B, C1);
        double t2 = run_and_time(n_runs, M, N, K, ikj_cpu_matmul, A, B, C2);
        double t3 = run_and_time(n_runs, M, N, K, scalar_sum_cpu_matmul, A, B, C3);

        std::cout << "Matrix size " << size << " x " << size << ":\n";
        std::cout << "  Naive (IJK)   : " << std::fixed << std::setprecision(6) << t1 << " sec\n";
        std::cout << "  IKJ         : " << t2 << " sec (match: " << (compare_matrices(C1, C2, M, N) ? "YES" : "NO") << ")\n";
        std::cout << "  Scalar Sum  : " << t3 << " sec (match: " << (compare_matrices(C1, C3, M, N) ? "YES" : "NO") << ")\n";

        for (int ts : tile_sizes) {
            double t4 = run_and_time(n_runs, M, N, K,
                [&](const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int m, int n, int k) {
                    tiled_cpu_matmul(a, b, c, m, n, k, ts);
                    //tiled_cpu_matmul_correct(a, b, c, m, n, k, ts);
                }, A, B, C4);

            bool correct = compare_matrices(C1, C4, M, N);
            std::cout << "  Tiled (" << std::setw(2) << ts << "): " << t4 << " sec (match: " << (correct ? "YES" : "NO") << ")\n";
        }

        std::cout << "\n";
    }

    return 0;
}
