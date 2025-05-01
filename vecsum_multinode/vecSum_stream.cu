#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define THREADS_PER_BLOCK 1024
#define NUM_BLOCK (256 * 1024)
#define NUM_DATA (THREADS_PER_BLOCK*NUM_BLOCK)
#define NUM_STREAMS 4

__global__ void vecAdd(int *_a, int *_b, int *_c, int _size) {
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    if (tID < _size)
        _c[tID] = _a[tID] + _b[tID];
}

int main(void)
{
    DS_timer timer(2);
    timer.setTimerName(0, "CUDA Total");
    timer.setTimerName(1, "VectorSum on Host");

    size_t memSize = sizeof(int) * NUM_DATA;
    printf("%zu elements, memSize = %zu bytes\n", NUM_DATA, memSize);

    // Allocate pinned host memory
    int *a, *b, *c, *h_c;
    cudaMallocHost(&a, memSize);
    cudaMallocHost(&b, memSize);
    cudaMallocHost(&c, memSize);
    h_c = (int *)malloc(memSize);

    for (int i = 0; i < NUM_DATA; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    timer.onTimer(1);
    for (int i = 0; i < NUM_DATA; i++)
        h_c[i] = a[i] + b[i];
    timer.offTimer(1);

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    size_t chunkSize = NUM_DATA / NUM_STREAMS;
    size_t chunkBytes = chunkSize * sizeof(int);
    //size_t numBlocksPerStream = (chunkSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaStream_t stream[NUM_STREAMS];
    cudaEvent_t start[NUM_STREAMS], end[NUM_STREAMS];


    // Setup streams, events, and per-stream device memory
    cudaMalloc(&d_a, NUM_DATA * sizeof(int));
    cudaMalloc(&d_b, NUM_DATA * sizeof(int));
    cudaMalloc(&d_c, NUM_DATA * sizeof(int));
   
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&stream[i]);
        cudaEventCreate(&start[i]);
        cudaEventCreate(&end[i]);
    }

    dim3 dimGrid((chunkSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 dimBlock(THREADS_PER_BLOCK);

    timer.onTimer(0);
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunkSize;
        //printf("Launching Stream[%d]\n",i);
        int deviceID;
        cudaGetDevice(&deviceID);
        printf("Launching Stream[%d] on Device[%d] with offset %d\n", i, deviceID, offset);

        cudaEventRecord(start[i], stream[i]);
        
        cudaMemcpyAsync(d_a + offset, a + offset, chunkBytes, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(d_b + offset, b + offset, chunkBytes, cudaMemcpyHostToDevice, stream[i]);

        vecAdd<<<dimGrid, dimBlock, 0, stream[i]>>>(d_a+offset, d_b+offset, d_c+offset, chunkSize);

        //cudaMemcpyAsync(c + offset, d_c[i], chunkBytes, cudaMemcpyDeviceToHost, stream[i]);
        cudaMemcpyAsync(c + offset, d_c+offset, chunkBytes, cudaMemcpyDeviceToHost, stream[i]);

        cudaEventRecord(end[i], stream[i]);
    }

    // Wait for all streams to complete
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(stream[i]);
    }

    timer.offTimer(0);
    timer.printTimer();

    //float totalStreamTime = 0.0f;
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaEventSynchronize(end[i]);
        float time = 0;
        cudaEventElapsedTime(&time, start[i], end[i]);
        //totalStreamTime += time;
        printf("Stream[%d] : %f ms\n", i, time);
    }
    //printf("Sum of all stream GPU times = %f ms\n", totalStreamTime);


    // Verify
    bool correct = true;
    for (int i = 0; i < NUM_DATA; i++) {
        if (h_c[i] != c[i]) {
            printf("Mismatch at %d: CPU = %d, GPU = %d\n", i, h_c[i], c[i]);
            correct = false;
            break;
        }
    }
    if (correct)
        printf("GPU works well!\n");

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        //cudaFree(d_a[i]);
        //cudaFree(d_b[i]);
        //cudaFree(d_c[i]);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaStreamDestroy(stream[i]);
    }

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    free(h_c);

    return 0;
}
