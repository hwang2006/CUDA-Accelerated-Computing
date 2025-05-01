// matmul_fp16_standard.cu
#include <cstdio>
#include <cuda_runtime.h>
#include "matmul.h" // Header declaring matmul_init, matmul, matmul_cleanup

// Error checking macro
#define CHECK_CUDA(call)    do {    cudaError_t status_ = call;    if (status_ != cudaSuccess) {      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_));      exit(EXIT_FAILURE);    }  } while (0)

// Block size for standard CUDA kernel
#define BLOCK_SIZE 32

// Kernel: Standard Matrix Multiplication
__global__ void matmul_kernel_standard(half *A, half *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += (float)(A[row * K + i] * B[i * N + col]);
        }
        C[row * N + col] = sum;
    }
}

// Static device pointers for A, B, and C
static half *A_gpu_standard, *B_gpu_standard;
static float *C_gpu_standard;

// Host function: Copy matrices to GPU and launch the standard kernel
void matmul_standard(half *_A, half *_B, float *_C, int M, int N, int K) {
    CHECK_CUDA(cudaMemcpy(A_gpu_standard, _A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_gpu_standard, _B, K * N * sizeof(half), cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel_standard<<<gridDim, blockDim>>>(A_gpu_standard, B_gpu_standard, _C, M, N, K);

    CHECK_CUDA(cudaGetLastError()); // Check kernel launch success
    CHECK_CUDA(cudaMemcpy(_C, C_gpu_standard, M * N * sizeof(float), cudaMemcpyDeviceToHost));
}

// Host function: Allocate memory on device for standard kernel
void matmul_init_standard(int M, int N, int K) {
    CHECK_CUDA(cudaMalloc(&A_gpu_standard, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&B_gpu_standard, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&C_gpu_standard, M * N * sizeof(float)));
}

// Host function: Free memory on device for standard kernel
void matmul_cleanup_standard(half *_A, half *_B, float *_C, int M, int N, int K) {
    CHECK_CUDA(cudaFree(A_gpu_standard));
    CHECK_CUDA(cudaFree(B_gpu_standard));
    CHECK_CUDA(cudaFree(C_gpu_standard));
}
