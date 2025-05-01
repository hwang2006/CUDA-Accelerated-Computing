#include <cuda.h>
#include <cuda_runtime.h>  // <-- Add this!
#include <cstdlib>
#include <cstring>
#include <cstdio>

#define NUM_DATA (256 * 64)
#define PTX_FILE "test1.ptx"
#define KERNEL_NAME "_Z6vecaddPfS_S_"

void kernelcall(float *d_a, float *d_b, float *d_c) {
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr a = (CUdeviceptr)d_a;
    CUdeviceptr b = (CUdeviceptr)d_b;
    CUdeviceptr c = (CUdeviceptr)d_c;

    // Initialize the CUDA driver API
    cuInit(0);

    CUcontext context;
    CUdevice device;
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // Load PTX
    cuModuleLoad(&module, PTX_FILE);
    cuModuleGetFunction(&kernel, module, KERNEL_NAME);

    // Set kernel args
    void* args[] = { &a, &b, &c };

    // Launch kernel
    cuLaunchKernel(kernel,
                   256, 1, 1,     // grid dim
                   256, 1, 1,     // block dim
                   0, 0, args, 0);

    cuCtxSynchronize();
}

int main() {
    float *d_a, *d_b, *d_c;
    int memSize = NUM_DATA * sizeof(float);

    float *a = (float*)malloc(memSize); memset(a, 0, memSize);
    float *b = (float*)malloc(memSize); memset(b, 0, memSize);
    float *c = (float*)malloc(memSize); memset(c, 0, memSize);

    for (int i = 0; i < NUM_DATA; i++) {
        a[i] = 1.0f;
        b[i] = i * 1.0f;
    }

    cudaMalloc(&d_a, memSize);
    cudaMalloc(&d_b, memSize);
    cudaMalloc(&d_c, memSize);

    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

    kernelcall(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);

    printf("First 5 elements: ");
    for (int i = 0; i < 5; i++) {
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
