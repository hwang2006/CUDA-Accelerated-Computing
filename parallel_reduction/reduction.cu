#include <cuda_runtime.h>
#include "reduction.hpp"

// Kernel code
__global__ void reduce_partial_sum(double* input, double* partial_sum, size_t num_elements) {
    extern __shared__ double cache[];  //cache[THREADS_PER_BLOCK];

    size_t tid = threadIdx.x;
    //size_t i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    size_t offset = 2 * blockIdx.x * blockDim.x;
    size_t stride = blockDim.x;
    size_t i = tid + offset;

    double sum = 0.0;
    if (i < num_elements) sum += input[i];
    //if (i + blockDim.x < num_elements) sum += input[i + blockDim.x];
    if (i + stride < num_elements) sum += input[i + stride];
    cache[tid] = sum;
    __syncthreads();

    // Parallel reduction within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sum[blockIdx.x] = cache[0];
    }
}

// Host code for reduction
double reduction_gpu(double* input_host, size_t num_elements) {
    size_t output_elements = (num_elements + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    double *input_gpu, *partial_sum_gpu;
    cudaMalloc(&input_gpu, num_elements * sizeof(double));
    cudaMalloc(&partial_sum_gpu, output_elements * sizeof(double));

    cudaMemcpy(input_gpu, input_host, num_elements * sizeof(double), cudaMemcpyHostToDevice);

    dim3 gridDim(output_elements);
    dim3 blockDim(THREADS_PER_BLOCK);
    reduce_partial_sum<<<gridDim, blockDim, THREADS_PER_BLOCK * sizeof(double), 0>>>(input_gpu, partial_sum_gpu, num_elements);

    // Allocate memory for partial sums on the host
    double* partial_sum = new double[output_elements];
    cudaMemcpy(partial_sum, partial_sum_gpu, output_elements * sizeof(double), cudaMemcpyDeviceToHost);

    // Final reduction on the host
    double result = 0.0;
    for (size_t i = 0; i < output_elements; ++i) {
        result += partial_sum[i];
    }

    delete[] partial_sum;
    cudaFree(input_gpu);
    cudaFree(partial_sum_gpu);
    return result;
}
