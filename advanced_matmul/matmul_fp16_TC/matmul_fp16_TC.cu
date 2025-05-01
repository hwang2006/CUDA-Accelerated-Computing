#include <cstdio>
#include <mma.h>    // For Tensor Core intrinsics (WMMA API)
#include "matmul.h" // Header declaring matmul_init, matmul, matmul_cleanup

using namespace nvcuda; // Needed for wmma namespace (tensor core ops)

// Error checking macro
#define CHECK_CUDA(call)   do {     cudaError_t status_ = call;     if (status_ != cudaSuccess) {       fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_));       exit(EXIT_FAILURE);     }   } while (0)

// Block and WMMA (Tensor Core tile) configuration
#define BLOCK_SIZE 32            // Tile size handled by one thread block
#define WMMA_M 16                // WMMA tile dimensions
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32             // 32 threads per warp
#define NUM_WARP ((WMMA_M * WMMA_N) / (WARP_SIZE))  // How many warps needed per block
#define C_LAYOUT wmma::mem_row_major  // Output matrix layout

// Kernel: Perform Matrix Multiplication using Tensor Cores
static __global__ void matmul_kernel(half *A, half *B, float *C, int M, int N, int K) {
  int gj = blockIdx.x;     // Block index along output matrix columns (N)
  int gi = blockIdx.y;     // Block index along output matrix rows (M)
  int lj = threadIdx.x;    // Local thread index in X
  int li = threadIdx.y;    // Local thread index in Y
  int warpId = li;         // Warp ID within the block (assuming li = warp index)

  // Early exit if block is outside matrix bounds
  if (gi * BLOCK_SIZE >= M || gj * BLOCK_SIZE >= N) return;

  // Shared memory to hold tiles of A and B matrices
  __shared__ half Alocal[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ half Blocal[BLOCK_SIZE * BLOCK_SIZE];

  // Tensor Core fragments (register memory)
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f); // Initialize accumulator to 0

  int A_row_index = gi * BLOCK_SIZE + li; // Base row index for A
  int B_col_index = gj * BLOCK_SIZE + lj; // Base column index for B

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

// Static device pointers for A, B, and C
static half *A_gpu, *B_gpu;
static float *C_gpu;

// Host function: Copy matrices to GPU and launch the kernel
void matmul(half *_A, half *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(cudaMemcpy(A_gpu, _A, M * K * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, _B, K * N * sizeof(half), cudaMemcpyHostToDevice));

  dim3 blockDim(BLOCK_SIZE, 4); // 32 threads along x, 4 along y
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  matmul_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);

  CHECK_CUDA(cudaGetLastError()); // Check kernel launch success
  CHECK_CUDA(cudaMemcpy(_C, C_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost));
}

// Host function: Allocate memory on device
void matmul_init(int M, int N, int K) {
  CHECK_CUDA(cudaMalloc(&A_gpu, M * K * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&B_gpu, K * N * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&C_gpu, M * N * sizeof(float)));
}

// Host function: Free memory on device
void matmul_cleanup(half *_A, half *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));
}
