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

#define M 16
#define N 16
#define K 16

#define BLOCK_SIZE 16
#define WARP_SIZE 32

__global__ void matmul_baseline(const float *A, const float *B, float *C, int width) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < width; ++i) {
        sum += A[row * width + i] * B[i * width + col];
    }
    C[row * width + col] = sum;
}

__global__ void matmul_tensorcore(const half *A, const half *B, float *C) {
    // Shared memory for A and B
    __shared__ half a_sh[M * K];
    __shared__ half b_sh[K * N];

    // Load A and B into shared memory
    int idx = threadIdx.x + threadIdx.y * blockDim.x; //idx = 0..31
    for (int offset = 0; offset < M * K / WARP_SIZE; offset++) {
       int index = idx + offset * WARP_SIZE; 
       if (index < M * K) a_sh[index] = A[index];
    }
 
    for (int offset = 0; offset < K * N / WARP_SIZE; offset++) {
       int index = idx + offset * WARP_SIZE;
       if (index < K * N) b_sh[index] = B[index];
    }

    //if (idx < M * K) a_sh[idx] = A[idx];
    //if (idx < K * N) b_sh[idx] = B[idx];

    __syncthreads();

    // Create fragments
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag;

    // load fragments from shared memory
    wmma::load_matrix_sync(a_frag, a_sh, K);
    wmma::load_matrix_sync(b_frag, b_sh, N);

    // load fragments from global memory
    //wmma::load_matrix_sync(a_frag, A, K);
    //wmma::load_matrix_sync(b_frag, B, N);

    wmma::fill_fragment(acc_frag, 0.0f);

    // Matrix multiply-accumulate using Tensor Cores
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // Store result to global memory
    wmma::store_matrix_sync(C, acc_frag, N, wmma::mem_row_major);
}

void generateMatrix(float *mat, int size, bool random = false) {
    for (int i = 0; i < size; ++i) {
        mat[i] = random ? (float)(rand() % 5) : 1.0f; // simple numbers
    }
}

void rand_mat(float *m, int R, int C) {
  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      m[i * C + j] = (float) rand() / RAND_MAX - 0.5;
    }
  }
}


bool verifyResult(const float *hostRef, const float *gpuRef, int size, float epsilon = 1e-1f) {
    for (int i = 0; i < size; ++i) {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("Mismatch at %d: host %f vs device %f\n", i, hostRef[i], gpuRef[i]);
            return false;
        }
    }
    return true;
}

int main() {
    float *h_A, *h_B, *h_C_baseline, *h_C_tensor;
    float *d_A_float, *d_B_float;
    half *d_A_half, *d_B_half;
    float *d_C_baseline, *d_C_tensor;

    int size = M * K;
    int bytes_f32 = size * sizeof(float);
    int bytes_f16 = size * sizeof(half);

    h_A = (float*)malloc(bytes_f32);
    h_B = (float*)malloc(bytes_f32);
    h_C_baseline = (float*)malloc(bytes_f32);
    h_C_tensor = (float*)malloc(bytes_f32);

    //generateMatrix(h_A, size, true);
    //generateMatrix(h_B, size, true);
    rand_mat(h_A, M, K);
    rand_mat(h_B, K, N);   

    cudaMalloc(&d_A_float, bytes_f32);
    cudaMalloc(&d_B_float, bytes_f32);
    cudaMalloc(&d_C_baseline, bytes_f32);

    cudaMemcpy(d_A_float, h_A, bytes_f32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_float, h_B, bytes_f32, cudaMemcpyHostToDevice);

    cudaMalloc(&d_A_half, bytes_f16);
    cudaMalloc(&d_B_half, bytes_f16);
    cudaMalloc(&d_C_tensor, bytes_f32);

    // Convert float -> half
    half *h_A_half = (half*)malloc(bytes_f16);
    half *h_B_half = (half*)malloc(bytes_f16);
    for (int i = 0; i < size; ++i) {
        h_A_half[i] = __float2half(h_A[i]);
        h_B_half[i] = __float2half(h_B[i]);
    }

    cudaMemcpy(d_A_half, h_A_half, bytes_f16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_half, h_B_half, bytes_f16, cudaMemcpyHostToDevice);

    DS_timer timer(2);
    timer.initTimers();
    timer.setTimerName(0, (char*)"Baseline (float32)");
    timer.setTimerName(1, (char*)"TensorCore (float16)");

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(1, 1);

    // Launch baseline
    timer.onTimer(0);
    matmul_baseline<<<blocks, threads>>>(d_A_float, d_B_float, d_C_baseline, K);
    cudaDeviceSynchronize();
    timer.offTimer(0);

    cudaMemcpy(h_C_baseline, d_C_baseline, bytes_f32, cudaMemcpyDeviceToHost);

    // Launch Tensor Core version
    timer.onTimer(1);
    matmul_tensorcore<<<1, dim3(32,1)>>>(d_A_half, d_B_half, d_C_tensor);
    cudaDeviceSynchronize();
    timer.offTimer(1);

    cudaMemcpy(h_C_tensor, d_C_tensor, bytes_f32, cudaMemcpyDeviceToHost);

    printf("\nVerification:\n");

    if (verifyResult(h_C_baseline, h_C_tensor, size))
        printf("TENSOR_CORE: Result is correct!\n");
    else
        printf("TENSOR_CORE: Result is NOT correct!\n");

    timer.printTimer();

    cudaFree(d_A_half);
    cudaFree(d_B_half);
    cudaFree(d_C_baseline);
    cudaFree(d_C_tensor);

    free(h_A);
    free(h_B);
    free(h_C_baseline);
    free(h_C_tensor);
    free(h_A_half);
    free(h_B_half);

    return 0;
}
