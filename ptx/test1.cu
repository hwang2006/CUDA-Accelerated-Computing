//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

__global__ void vecadd(float *a, float *b, float *c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] + b[i];
}

void kernelcall(float *a, float *b, float *c) {
    vecadd<<<256, 256>>>(a, b, c);
}

