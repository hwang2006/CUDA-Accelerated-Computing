/*
 * A100: -arch=sm_80,  V100: -arch=sm_70, H100: -arch=sm_90
 * $ nvcc test_TC.cu ../../DSTimer/DS_timer.cpp -I../../DSTimer -arch=sm_80 -o test_TC 
 */

#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include "DS_timer.h"

using namespace nvcuda;

#define M 32
#define N 32
//#define K 128
#define K 100000

#define BLOCK_SIZE 32
#define WARP_SIZE 32

#define WMMA_M 16                // WMMA tile dimensions
#define WMMA_N 16
#define WMMA_K 16
#define NUM_WARP ((WMMA_M * WMMA_N) / (WARP_SIZE))  // How many warps needed per block
#define C_LAYOUT wmma::mem_row_major  // Output matrix layout

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void matmul_baseline(const float *A, const float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

__global__ void matmul_tensorcore(const half *A, const half *B, float *C) {
  //int gj = blockIdx.x;     // Block index along output matrix columns (N)
  //int gi = blockIdx.y;     // Block index along output matrix rows (M)
  int lj = threadIdx.x;    // Local thread index in X
  int li = threadIdx.y;    // Local thread index in Y
  int warpId = li;         // Warp ID within the block (assuming li = warp index)

  // Early exit if block is outside matrix bounds
  //if (gi * BLOCK_SIZE >= M || gj * BLOCK_SIZE >= N) return;

  // Shared memory to hold tiles of A and B matrices
  __shared__ half Alocal[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ half Blocal[BLOCK_SIZE * BLOCK_SIZE];

  // Tensor Core fragments (register memory)
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f); // Initialize accumulator to 0

  //int A_row_index = gi * BLOCK_SIZE + li; // Base row index for A
  int A_row_index = li; // Base row index for A
  //int B_col_index = gj * BLOCK_SIZE + lj; // Base column index for B
  int B_col_index = lj; // Base column index for B

  // Loop over K dimension tiles 
  for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
    // Cooperative loading of tiles into shared memory
    for (int offset = 0; offset < NUM_WARP; ++offset) {
      int A_col_index = bk + lj; //blockDim.y == 4 
      Alocal[(li + offset * blockDim.y) * BLOCK_SIZE + lj] =
        ((A_row_index + offset * blockDim.y) < M && A_col_index < K)
          ? A[(A_row_index + offset * blockDim.y) * K + A_col_index]
          : (half)(0.0); // Zero padding if out-of-bounds

      int B_row_index = bk + li + (offset * blockDim.y);
      Blocal[(li + offset * blockDim.y) * BLOCK_SIZE + lj] =
        (B_row_index < K && B_col_index < N)
          ? B[B_row_index * N + B_col_index]
          : (half)(0.0); // Zero padding if out-of-bounds
    }

    __syncthreads(); // Wait until all threads finish loading shared memory

    // Perform MMA (matrix multiply-accumulate) over the loaded tiles
    for (int i = 0; i < BLOCK_SIZE; i += WMMA_K) {
      int aCol = i;                      // Subtile column offset for A
      int aRow = (warpId / 2) * WMMA_M;   // Subtile row offset for A
      int bCol = (warpId % 2) * WMMA_N;   // Subtile column offset for B
      int bRow = i;                      // Subtile row offset for B

      // Load WMMA fragments from shared memory
      wmma::load_matrix_sync(a_frag, Alocal + aCol + aRow * BLOCK_SIZE, BLOCK_SIZE);
      wmma::load_matrix_sync(b_frag, Blocal + bCol + bRow * BLOCK_SIZE, BLOCK_SIZE);

      // Tensor core multiply-accumulate
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __syncthreads(); // Wait until all threads finish computation before next tile
  }

  // Compute final coordinates in C matrix
  int cRow = (warpId / 2) * WMMA_M + blockIdx.y * blockDim.y * NUM_WARP;
  int cCol = (warpId % 2) * WMMA_N + blockIdx.x * blockDim.x;

  // Store the result fragment back to global memory
  if (cRow + WMMA_M <= M && cCol + WMMA_N <= N) {
    wmma::store_matrix_sync(C + cCol + cRow * N, c_frag, N, C_LAYOUT);
  }
}

void generateMatrix(float *mat, int size, bool random = false) {
    for (int i = 0; i < size; ++i) {
        mat[i] = random ? (float)(rand() % 5) : 1.0f; // simple numbers
    }
}

half *alloc_mat_half(int R, int C) {
  half *m;
  CHECK_CUDA(cudaMalloc(&m, sizeof(half) * R * C));
  return m;
}

float *alloc_mat_float(int R, int C) {
  float *m;
  CHECK_CUDA(cudaMalloc(&m, sizeof(float) * R * C));
  return m;
}

void rand_mat(float *m, int R, int C) {
  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      m[i * C + j] = (float) rand() / RAND_MAX - 0.5;
    }
  }
}


bool verifyResult(const float *hostRef, const float *gpuRef, int size, float epsilon = 1e-3f) {
    for (int i = 0; i < size; ++i) {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("Mismatch at %d: host %f vs device %f\n", i, hostRef[i], gpuRef[i]);
            return false;
        }
    }
    return true;
}

int main() {
    float *h_A, *h_B, *h_C, *h_C_baseline, *h_C_tensor;
    float *d_A_float, *d_B_float;
    half *d_A_half, *d_B_half;
    float *d_C_baseline, *d_C_tensor;

    //int size = M * K;
    //int bytes_f32 = size * sizeof(float);
    //int bytes_f16 = size * sizeof(half);

    h_A = (float*)malloc(sizeof(float) * M * K);
    h_B = (float*)malloc(sizeof(float) * K * N);
    h_C = (float*)malloc(sizeof(float) * M * N);
    //h_C_mixed = (float*)malloc(sizeof(float) * M * N);
    h_C_baseline = (float*)malloc(sizeof(float) * M * N);
    h_C_tensor = (float*)malloc(sizeof(float) * M * N);

    //generateMatrix(h_A, size, true);
    //generateMatrix(h_B, size, true);
    rand_mat(h_A, M, K);
    rand_mat(h_B, K, N);   

    d_A_float = alloc_mat_float(M,K);   
    d_B_float = alloc_mat_float(K,N);   
    d_C_baseline = alloc_mat_float(M,N);   
    
    CHECK_CUDA(cudaMemcpy(d_A_float, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_float, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice));

    d_A_half = alloc_mat_half(M,K);
    d_B_half = alloc_mat_half(K,N);
    d_C_tensor = alloc_mat_float(M,N);

    // Convert float -> half
    half *h_A_half = (half*)malloc(sizeof(half) * M * K);
    half *h_B_half = (half*)malloc(sizeof(half) * K * N);

    for (int i = 0; i < M * K; ++i) {
        h_A_half[i] = __float2half(h_A[i]);
    }

    for (int i = 0; i < K * N; ++i) {
        h_B_half[i] = __float2half(h_B[i]);
    }

    memset(h_C, 0, sizeof(float) * M * N);
#pragma omp parallel for num_threads(20)
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
          h_C[i * N + j] += h_A[i * K + k] * h_B[k * N + j];
        }
      }
    }


    CHECK_CUDA(cudaMemcpy(d_A_half, h_A_half, sizeof(half) * M * K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_half, h_B_half, sizeof(half) * K * N, cudaMemcpyHostToDevice));

    DS_timer timer(2);
    timer.initTimers();
    timer.setTimerName(0, (char*)"Baseline (float32)");
    timer.setTimerName(1, (char*)"TensorCore (float16)");

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE ); //(1,1)
    dim3 gridDim(1,1);

    // Launch baseline
    timer.onTimer(0);
    matmul_baseline<<<gridDim, blockDim>>>(d_A_float, d_B_float, d_C_baseline);
    //matmul_baseline<<<1, blockDim>>>(d_A_float, d_B_float, d_C_baseline);
    cudaDeviceSynchronize();
    timer.offTimer(0);

    CHECK_CUDA(cudaMemcpy(h_C_baseline, d_C_baseline, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

   
    dim3 threads(BLOCK_SIZE, 4);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE); //(1,1)
    
    // Launch Tensor Core version
    timer.onTimer(1);
    matmul_tensorcore<<<blocks, threads>>>(d_A_half, d_B_half, d_C_tensor);
    cudaDeviceSynchronize();
    timer.offTimer(1);

    CHECK_CUDA(cudaMemcpy(h_C_tensor, d_C_tensor, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    printf("\nVerification:\n");

    if (verifyResult(h_C, h_C_baseline, M * N))
        printf("Baseline (float32): Result is correct!\n");
    else
        printf("Baseline (float32): Result is NOT correct!\n");

    //if (verifyResult(h_C, h_C_tensor, M * N, 1e-1f)) //epsilon = 1e-1f
    if (verifyResult(h_C_baseline, h_C_tensor, M * N, 1e-1f)) //epsilon = 1e-1f
        printf("TENSOR_CORE (fp16 mixed precision): Result is correct!\n");
    else
        printf("TENSOR_CORE (fp16 mixed precision): Result is NOT correct!\n");

    timer.printTimer();

    CHECK_CUDA(cudaFree(d_A_half));
    CHECK_CUDA(cudaFree(d_A_float));
    CHECK_CUDA(cudaFree(d_B_half));
    CHECK_CUDA(cudaFree(d_B_float));
    CHECK_CUDA(cudaFree(d_C_baseline));
    CHECK_CUDA(cudaFree(d_C_tensor));
    CHECK_CUDA(cudaDeviceSynchronize());

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_baseline);
    free(h_C_tensor);
    free(h_A_half);
    free(h_B_half);

    return 0;
}
