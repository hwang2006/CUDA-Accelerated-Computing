#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define THREADS_PER_BLOCK 1024
#define NUM_BLOCK (1024 * 1024)
#define NUM_DATA (THREADS_PER_BLOCK * NUM_BLOCK)
#define NUM_STREAMS_PER_GPU 4

__global__ void vecAdd(int *_a, int *_b, int *_c, int _size) {
    //int tID = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tID = blockIdx.x * blockDim.x + threadIdx.x;
    if (tID < _size)
        _c[tID] = _a[tID] + _b[tID];
}

int main(void)
{
    DS_timer timer(2);
    timer.setTimerName(0, "CUDA Total");
    timer.setTimerName(1, "VectorSum on Host");

    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0) {
        printf("No CUDA-capable GPU found.\n");
        return -1;
    }

    size_t memSize = sizeof(int) * NUM_DATA;
    printf("%zu elements, memSize = %zu bytes, using %d GPUs\n", NUM_DATA, memSize, numDevices);

    // Host allocations (pinned)
    int *a, *b, *c, *h_c;
    cudaMallocHost(&a, memSize);
    cudaMallocHost(&b, memSize);
    cudaMallocHost(&c, memSize);
    h_c = (int *)malloc(memSize);

    //for (int i = 0; i < NUM_DATA; i++) {
    for (size_t i = 0; i < NUM_DATA; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    // CPU reference
    timer.onTimer(1);
    //for (int i = 0; i < NUM_DATA; i++)
    for (size_t i = 0; i < NUM_DATA; i++)
        h_c[i] = a[i] + b[i];
    timer.offTimer(1);

    // Divide work across GPUs
    size_t chunkPerGPU = NUM_DATA / numDevices;
    size_t chunkPerStream = chunkPerGPU / NUM_STREAMS_PER_GPU;
    size_t chunkBytes = chunkPerStream * sizeof(int);
    dim3 dimBlock(THREADS_PER_BLOCK);
    dim3 dimGrid((chunkPerStream + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    timer.onTimer(0);

    for (int dev = 0; dev < numDevices; dev++) {
        cudaSetDevice(dev);

        // Allocate per-GPU memory
        int *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, chunkPerGPU * sizeof(int));
        cudaMalloc(&d_b, chunkPerGPU * sizeof(int));
        cudaMalloc(&d_c, chunkPerGPU * sizeof(int));

        cudaStream_t streams[NUM_STREAMS_PER_GPU];
        cudaEvent_t start[NUM_STREAMS_PER_GPU], end[NUM_STREAMS_PER_GPU];

        for (int s = 0; s < NUM_STREAMS_PER_GPU; s++) {
            cudaStreamCreate(&streams[s]);
            cudaEventCreate(&start[s]);
            cudaEventCreate(&end[s]);

            //int globalOffset = dev * chunkPerGPU + s * chunkPerStream;
            size_t globalOffset = dev * chunkPerGPU + s * chunkPerStream;

            cudaEventRecord(start[s], streams[s]);
            cudaMemcpyAsync(d_a + s * chunkPerStream, a + globalOffset, chunkBytes, cudaMemcpyHostToDevice, streams[s]);
            cudaMemcpyAsync(d_b + s * chunkPerStream, b + globalOffset, chunkBytes, cudaMemcpyHostToDevice, streams[s]);

            vecAdd<<<dimGrid, dimBlock, 0, streams[s]>>>(
                d_a + s * chunkPerStream,
                d_b + s * chunkPerStream,
                d_c + s * chunkPerStream,
                chunkPerStream
            );

            cudaMemcpyAsync(c + globalOffset, d_c + s * chunkPerStream, chunkBytes, cudaMemcpyDeviceToHost, streams[s]);
            cudaEventRecord(end[s], streams[s]);
        }

        for (int s = 0; s < NUM_STREAMS_PER_GPU; s++) {
            cudaStreamSynchronize(streams[s]);
            float time = 0;
            cudaEventElapsedTime(&time, start[s], end[s]);
            printf("Device[%d] Stream[%d]: %.2f ms\n", dev, s, time);
            cudaEventDestroy(start[s]);
            cudaEventDestroy(end[s]);
            cudaStreamDestroy(streams[s]);
        }

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    timer.offTimer(0);
    timer.printTimer();

    // Verify
    bool correct = true;
    //for (int i = 0; i < NUM_DATA; i++) {
    for (size_t i = 0; i < NUM_DATA; i++) {
        if (h_c[i] != c[i]) {
            //printf("Mismatch at %d: CPU = %d, GPU = %d\n", i, h_c[i], c[i]);
            printf("Mismatch at %zu: CPU = %d, GPU = %d\n", i, h_c[i], c[i]);
            correct = false;
            break;
        }
    }
    if (correct)
        printf("GPU works well across %d devices!\n", numDevices);

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    free(h_c);

    return 0;
}
