#include <cstdlib>
#include <cstdio>
#include "convolution.cuh"

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
#define NUM_WARP 32
#define BLOCK_PER_WARP (MAX_BLOCK / NUM_WARP)
#define BLOCK_SIZE 32
// Device(GPU) pointers
static float *I_gpu, *F_gpu, *O_gpu, *BUF1_gpu, *BUF2_gpu;

__global__ void im2col_kernel(float *_I, float *workspace, int N, int C, int H, int W,
                              int R, int S, int pad_h, int pad_w, int stride_h,
                              int stride_w, int dilation_h, int dilation_w, int OH, int OW){

  int tidx = threadIdx.x + blockIdx.x * blockDim.x;

  for(int crs = 0 ; crs < C * R * S ; ++crs){
    int row_offset = crs * N * OH * OW;
    const int c = crs / (R * S);
    const int r = (crs / S) % R;
    const int s = crs % S;

    for(int nhw = tidx ; nhw < N * OH * OW ; nhw += MAX_BLOCK * NUM_SM){
      const int n = nhw / (OH * OW);
      const int oh = (nhw / OW) % OH;
      const int ow = nhw % OW;
      
      const int h = oh * stride_h - pad_h + r * dilation_h;
      const int w = ow * stride_w - pad_w + s * dilation_w;

      if (h < 0 || h >= H || w < 0 || w >= W) continue;

      workspace[row_offset + nhw] =
      _I[n * C * H * W + c * H * W + h * W + w];

    }
  }
}

static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                                     int K) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int gj = blockIdx.x;
  int gi = blockIdx.y;
  if (gi * BLOCK_SIZE >= M || gj * BLOCK_SIZE >= N) return;  // boundary check
  int lj = threadIdx.x;
  int li = threadIdx.y;
  __shared__ float Alocal[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Blocal[BLOCK_SIZE][BLOCK_SIZE];
  float c = 0.f;
  int A_row_index = (gi * BLOCK_SIZE + li);
  int B_col_index = (gj * BLOCK_SIZE + lj);
  for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
    int A_col_index = bk + lj;
    Alocal[li][lj] = (A_row_index < M && A_col_index < K)
                         ? A[A_row_index * K + A_col_index]
                         : 0.f;
    int B_row_index = bk + li;
    Blocal[li][lj] = (B_row_index < K && B_col_index < N)
                         ? B[B_row_index * N + B_col_index]
                         : 0.f;
    __syncthreads();
    for (int lk = 0; lk < BLOCK_SIZE; ++lk) {
      c += Alocal[li][lk] * Blocal[lk][lj];
    }
    __syncthreads();
  }
  if (i < M && j < N) C[i * N + j] = c;
}

__global__ void reshape_kernel(float *_src, float *_dst, int N, int K, int OH, int OW){

  int bidx = blockIdx.x;
  int widx = threadIdx.x / NUM_WARP;
  int lidx = threadIdx.x % NUM_WARP;

  for(int k = widx ; k < K ; k += BLOCK_PER_WARP){
    for(int on = bidx ; on < N ; on += NUM_SM){
      for(int hw = lidx ; hw < OH * OW ; hw += NUM_WARP){
        _dst[on * K * OH * OW + k * OH * OW + hw] = 
          _src[k * N * OH * OW + on * OH * OW + hw];
      }
    }
  }
}

void convolution_im2col(float *_I, float *_F, float *_O, float *_BUF1,
                                  float *_BUF2, int N, int C, int H, int W,
                                  int K, int R, int S, int pad_h, int pad_w,
                                  int stride_h, int stride_w, int dilation_h,
                                  int dilation_w) {

  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  CHECK_CUDA(
      cudaMemcpy(I_gpu, _I, sizeof(float) * N * C * H * W, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(F_gpu, _F, sizeof(float) * K * C * R * S, cudaMemcpyHostToDevice));

  dim3 griddim_im2col(NUM_SM);
  dim3 blockdim_im2col(MAX_BLOCK);
  im2col_kernel<<<griddim_im2col, blockdim_im2col>>>(I_gpu, BUF1_gpu, N, C, H, W, R, S, 
                                      pad_h, pad_w, stride_h, stride_w,
                                      dilation_h, dilation_w, OH, OW);
  CHECK_CUDA(cudaGetLastError());

  dim3 griddim_matmul(((N * OH * OW)+ BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 blockDim_matmul(BLOCK_SIZE, BLOCK_SIZE);
  matmul_kernel<<<griddim_matmul, blockDim_matmul>>>(F_gpu, BUF1_gpu, BUF2_gpu, K, N * OH * OW, C * R * S);
  CHECK_CUDA(cudaGetLastError());

  dim3 griddim_reshape(NUM_SM);
  dim3 blockDim_reshape(MAX_BLOCK);
  reshape_kernel<<<griddim_reshape, blockDim_reshape>>>(BUF2_gpu, O_gpu, N, K, OH, OW);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(
    cudaMemcpy(_O, O_gpu, sizeof(float) * N * K * OH * OW, cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution(float *_I, float *_F, float *_O, float *_BUF1, float *_BUF2,
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

  size_t alloc = sizeof(float) * N * C * H * W + sizeof(float) * K * C * R * S + sizeof(float) * N * K * OH * OW + 
  sizeof(float) * C * R * S * N * OH * OW + sizeof(float) * N * K * OH * OW;
  printf("Alloc Memory : %lf\n",(double)alloc/1e9);

  CHECK_CUDA(cudaMalloc((void **) &I_gpu, sizeof(float) * N * C * H * W));
  CHECK_CUDA(cudaMalloc((void **) &F_gpu, sizeof(float) * K * C * R * S));
  CHECK_CUDA(cudaMalloc((void **) &O_gpu, sizeof(float) * N * K * OH * OW));
  CHECK_CUDA(cudaMalloc((void **) &BUF1_gpu, sizeof(float) * C * R * S * N * OH * OW));
  CHECK_CUDA(cudaMalloc((void **) &BUF2_gpu, sizeof(float) * N * K * OH * OW));
}

void convolution_cleanup(float *_I, float *_F, float *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w) {
  CHECK_CUDA(cudaFree(I_gpu));
  CHECK_CUDA(cudaFree(F_gpu));
  CHECK_CUDA(cudaFree(O_gpu));
  CHECK_CUDA(cudaFree(BUF1_gpu));
  CHECK_CUDA(cudaFree(BUF2_gpu));
}
