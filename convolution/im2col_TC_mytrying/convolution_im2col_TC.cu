#include <cstdlib>
#include <cstdio>
#include <mma.h>
#include "convolution.cuh"

using namespace nvcuda;


#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define MAX_BLOCK 1024
#define NUM_SM 80
#define WARP_SIZE 32
#define BLOCK_PER_WARP (MAX_BLOCK / WARP_SIZE)
#define BLOCK_SIZE 32

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define NUM_WARP ((WMMA_M * WMMA_N) / (WARP_SIZE))
#define C_LAYOUT wmma::mem_row_major
// Device(GPU) pointers
static half *I_gpu, *F_gpu, *BUF1_gpu;
static float *O_gpu, *BUF2_gpu;

__global__ void im2col_kernel(half *_I, half *workspace, int N, int C, int H, int W,
                              int R, int S, int pad_h, int pad_w, int stride_h,
                              int stride_w, int dilation_h, int dilation_w, int OH, int OW){

  int tidx = threadIdx.x + blockIdx.x * blockDim.x;

  for(int crs = 0 ; crs < C * R * S ; ++crs){
    int row_offset = (size_t)crs * N * OH * OW;
    const int c = crs / (R * S);
    const int r = (crs / S) % R;
    const int s = crs % S;

    for(size_t nhw = tidx ; nhw < N * OH * OW ; nhw += MAX_BLOCK * NUM_SM){
      const int n = nhw / (OH * OW);
      const int oh = (nhw / OW) % OH;
      const int ow = nhw % OW;
      
      const int h = oh * stride_h - pad_h + r * dilation_h;
      const int w = ow * stride_w - pad_w + s * dilation_w;

      if (h < 0 || h >= H || w < 0 || w >= W) continue;

      workspace[row_offset + nhw] =
      _I[(size_t)n * C * H * W + c * H * W + h * W + w];

    }
  }
}

static __global__ void matmul_kernel(half *A, half *B, float *C, int M, int N,
                                     int K) {
  int gj = blockIdx.x;
  int gi = blockIdx.y;
  if (gi * BLOCK_SIZE >= M || gj * BLOCK_SIZE >= N) return;  // boundary check
  int lj = threadIdx.x;
  int li = threadIdx.y;
  int warpId = li;

  __shared__ half Alocal[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ half Blocal[BLOCK_SIZE * BLOCK_SIZE];

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  int A_row_index = (gi * BLOCK_SIZE + li);
  int B_col_index = (gj * BLOCK_SIZE + lj);

  for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
    
    for(int offset = 0 ; offset < NUM_WARP ; ++offset){
      int A_col_index = bk + lj;
      Alocal[(li + offset * blockDim.y) * BLOCK_SIZE + lj] = 
        ((A_row_index + offset * blockDim.y) < M && A_col_index < K)
        ? A[((size_t)A_row_index + offset * blockDim.y) * K + A_col_index]
        : (half)(0.0);

      int B_row_index = bk + li + (offset * blockDim.y);
      Blocal[(li + offset * blockDim.y) * BLOCK_SIZE + lj] = 
      (B_row_index < K && B_col_index < N)
        ? B[(size_t)B_row_index * N + B_col_index]
        : (half)(0.0);  
    }
    __syncthreads();

    for (int i = 0; i < BLOCK_SIZE; i += WMMA_K) {
      int aCol = i;
      int aRow = (warpId / 2) * WMMA_M;
      int bCol = (warpId % 2) * WMMA_N;
      int bRow = i;

      wmma::load_matrix_sync(a_frag, Alocal + aCol + aRow * BLOCK_SIZE, BLOCK_SIZE);
      wmma::load_matrix_sync(b_frag, Blocal + bCol + bRow * BLOCK_SIZE, BLOCK_SIZE);
      
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __syncthreads();
  }

  int cRow = (warpId / 2) * WMMA_M + blockIdx.y * blockDim.y * NUM_WARP;
  int cCol = (warpId % 2) * WMMA_N + blockIdx.x * blockDim.x;

  if(cRow + WMMA_M <= M && cCol + WMMA_N <= N){
    wmma::store_matrix_sync(C + ((size_t)cCol + cRow * N), c_frag, N, C_LAYOUT);
  }
}

__global__ void reshape_kernel(float *_src, float *_dst, int N, int K, int OH, int OW){

  int bidx = blockIdx.x;
  int widx = threadIdx.x / WARP_SIZE;
  int lidx = threadIdx.x % WARP_SIZE;

  for(int k = widx ; k < K ; k += BLOCK_PER_WARP){
    for(int on = bidx ; on < N ; on += NUM_SM){
      for(int hw = lidx ; hw < OH * OW ; hw += WARP_SIZE){
        _dst[on * K * OH * OW + k * OH * OW + hw] = 
          _src[k * N * OH * OW + on * OH * OW + hw];

      }
    }
  }
}

void convolution_im2col(half *_I, half *_F, float *_O, half *_BUF1,
                                  float *_BUF2, int N, int C, int H, int W,
                                  int K, int R, int S, int pad_h, int pad_w,
                                  int stride_h, int stride_w, int dilation_h,
                                  int dilation_w) {

  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  printf("\nDimension of GEMM : M[%d], N[%d], K[%d]\n",K, N * OH * OW, C * R * S);

  CHECK_CUDA(
      cudaMemcpy(I_gpu, _I, sizeof(half) * (size_t)N * C * H * W, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(F_gpu, _F, sizeof(half) * (size_t)K * C * R * S, cudaMemcpyHostToDevice));

  dim3 griddim_im2col(NUM_SM);
  dim3 blockdim_im2col(MAX_BLOCK);
  im2col_kernel<<<griddim_im2col, blockdim_im2col>>>(I_gpu, BUF1_gpu, N, C, H, W, R, S, 
                                      pad_h, pad_w, stride_h, stride_w,
                                      dilation_h, dilation_w, OH, OW);
  CHECK_CUDA(cudaGetLastError());
  
  dim3 griddim_matmul(((N * OH * OW) + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 blockDim_matmul(BLOCK_SIZE, 4);
  matmul_kernel<<<griddim_matmul, blockDim_matmul>>>(F_gpu, BUF1_gpu, BUF2_gpu, K, N * OH * OW, C * R * S);
  CHECK_CUDA(cudaGetLastError());

  dim3 griddim_reshape(NUM_SM);
  dim3 blockDim_reshape(MAX_BLOCK);
  reshape_kernel<<<griddim_reshape, blockDim_reshape>>>(BUF2_gpu, O_gpu, N, K, OH, OW);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(
    cudaMemcpy(_O, O_gpu, sizeof(float) * (size_t)N * K * OH * OW, cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution(half *_I, half *_F, float *_O, half *_BUF1, float *_BUF2,
                 int N, int C, int H, int W, int K, int R, int S, int pad_h,
                 int pad_w, int stride_h, int stride_w, int dilation_h,
                 int dilation_w) {
  // Remove this line after you complete the convolution on GPU
  
  convolution_im2col(_I, _F, _O, _BUF1, _BUF2, N, C, H, W, K, R, S,
                               pad_h, pad_w, stride_h, stride_w, dilation_h,
                               dilation_w);
}

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w) {
  
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  size_t alloc = sizeof(half) * (size_t)N * C * H * W + sizeof(half) * (size_t)K * C * R * S + sizeof(float) * (size_t)N * K * OH * OW + 
  sizeof(half) * (size_t)C * R * S * N * OH * OW + sizeof(float) * (size_t)N * K * OH * OW;
  printf("GPU Memory Alloc : %lf GB\n",(double)alloc/1e9);

  CHECK_CUDA(cudaMalloc((void **) &I_gpu, sizeof(half) * N * C * H * W));
  CHECK_CUDA(cudaMalloc((void **) &F_gpu, sizeof(half) * K * C * R * S));
  CHECK_CUDA(cudaMalloc((void **) &O_gpu, sizeof(float) * N * K * OH * OW));
  CHECK_CUDA(cudaMalloc((void **) &BUF1_gpu, sizeof(half) * C * R * S * N * OH * OW));
  CHECK_CUDA(cudaMalloc((void **) &BUF2_gpu, sizeof(float) * N * K * OH * OW));
}

void convolution_cleanup(half *_I, half *_F, float *_O, half* BUF1_, float* BUF2_, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w) {
  CHECK_CUDA(cudaFree(I_gpu));
  CHECK_CUDA(cudaFree(F_gpu));
  CHECK_CUDA(cudaFree(O_gpu));
  CHECK_CUDA(cudaFree(BUF1_gpu));
  CHECK_CUDA(cudaFree(BUF2_gpu));
}
