#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdlib>
#include <cstring>
#include <cstdio>

#define NUM_DATA (256 * 64)

void kernelcall(float *a, float *b, float *c);

int main(){
  float *d_a, *d_b, *d_c;
  int memSize = NUM_DATA * sizeof(float);

  float *a = (float*)malloc(memSize); memset(a, 0, memSize);
  float *b = (float*)malloc(memSize); memset(b, 0, memSize);
  float *c = (float*)malloc(memSize); memset(c, 0, memSize);

  for (int i = 0; i < NUM_DATA; i++){
      a[i] = 1.0f;
      b[i] = i*1.0f;
  }

  cudaMalloc(&d_a, memSize);
  cudaMalloc(&d_b, memSize);
  cudaMalloc(&d_c, memSize);

  cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

  kernelcall(d_a, d_b, d_c);

  cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);

  printf("First 5 elements: ");
  for(int i = 0; i < 5; i++){
    printf("%.2f ", c[i]);
  }
  printf("\n");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(a);
  free(b);
  free(c);

  return 0;
}

