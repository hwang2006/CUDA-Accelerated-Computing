#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <DS_timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define LOOP_I(_loop) for(int i=0; i < _loop; i++)

#define NUM_BLOCK (128*1024)
#define NUM_T_IN_B 1024
#define ARRAY_SIZE (NUM_T_IN_B*NUM_BLOCK)
#define NUM_STREAMS 4

__global__ void myKernel2(int *_in, int *_out)
{
    int tID = blockDim.x * blockIdx.x + threadIdx.x;
    int temp = 0;
    for (int i = 0; i < 500; i++) {
        temp = (temp + _in[tID] * 5) % 10;
    }
    _out[tID] = temp;
}

void runTest(size_t array_size, bool use_pinned, const char* label) {
    printf("========== %s Memory (Size: %zu bytes) =========="
           "\n", label, array_size);

    int *in = NULL, *out = NULL, *out2 = NULL;
    if (use_pinned) {
        cudaMallocHost(&in, sizeof(int)*array_size);
        cudaMallocHost(&out, sizeof(int)*array_size);
        cudaMallocHost(&out2, sizeof(int)*array_size);
    } else {
        in = (int*)malloc(sizeof(int)*array_size);
        out = (int*)malloc(sizeof(int)*array_size);
        out2 = (int*)malloc(sizeof(int)*array_size);
    }

    memset(in, 0, sizeof(int)*array_size);
    memset(out, 0, sizeof(int)*array_size);
    memset(out2, 0, sizeof(int)*array_size);

    LOOP_I(array_size) in[i] = rand() % 10;

    int *dIn, *dOut;
    cudaMalloc(&dIn, sizeof(int)*array_size);
    cudaMalloc(&dOut, sizeof(int)*array_size);

    cudaStream_t stream[NUM_STREAMS];
    cudaEvent_t start[NUM_STREAMS], end[NUM_STREAMS];

    LOOP_I(NUM_STREAMS) {
        cudaStreamCreate(&stream[i]);
        cudaEventCreate(&start[i]); cudaEventCreate(&end[i]);
    }

    int chunkSize = array_size / NUM_STREAMS;

    LOOP_I(NUM_STREAMS) {
        int offset = chunkSize * i;
        cudaEventRecord(start[i], stream[i]);
        cudaMemcpyAsync(dIn + offset, in + offset, sizeof(int)*chunkSize, cudaMemcpyHostToDevice, stream[i]);
        myKernel2 <<<chunkSize / NUM_T_IN_B, NUM_T_IN_B, 0, stream[i]>>> (dIn + offset, dOut + offset);
        cudaMemcpyAsync(out2 + offset, dOut + offset, sizeof(int)*chunkSize, cudaMemcpyDeviceToHost, stream[i]);
        cudaEventRecord(end[i], stream[i]);
    }

    cudaDeviceSynchronize();

    LOOP_I(NUM_STREAMS) {
        float time = 0;
        cudaEventElapsedTime(&time, start[i], end[i]);
        double throughput = (sizeof(int)*chunkSize) / (time / 1000.0) / (1 << 30);
        printf("Stream[%d] : %6.2f ms | Throughput: %6.2f GB/s\n", i, time, throughput);
    }

    LOOP_I(array_size) {
        if (out[i] != out2[i]) {
            printf("Data mismatch at index %d\n", i);
            break;
        }
    }

    LOOP_I(NUM_STREAMS) {
        cudaStreamDestroy(stream[i]);
        cudaEventDestroy(start[i]);
        cudaEventDestroy(end[i]);
    }

    cudaFree(dIn);
    cudaFree(dOut);

    if (use_pinned) {
        cudaFreeHost(in);
        cudaFreeHost(out);
        cudaFreeHost(out2);
    } else {
        free(in);
        free(out);
        free(out2);
    }
}

int main(void) {
    std::vector<size_t> sizes = {
        1024, 4096, 16384, 65536, 262144, 1048576,
        4194304, 16777216, 67108864, 268435456
    };

    for (size_t size : sizes) {
        runTest(size, false, "Pageable");
        runTest(size, true,  "Pinned");
        printf("\n");
    }
    return 0;
}
