#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <DS_timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOOP_I(_loop) for(int i=0; i < _loop; i++)

//#define NUM_BLOCK (128*1024)
#define NUM_BLOCK (16 * 1024)
#define NUM_T_IN_B 1024
#define ARRAY_SIZE (NUM_T_IN_B*NUM_BLOCK)

#define NUM_STREAMS 4
//#define NUM_STREAMS 1

__global__ void myKernel2(int *_in, int *_out)
{
    int tID = blockDim.x * blockIdx.x + threadIdx.x;

    int temp = 0;
    for (int i = 0; i < 250; i++) {
        temp = (temp + _in[tID] * 5) % 10;
    }
    _out[tID] = temp;
}

int main(void)
{
    DS_timer timer(2);
    timer.setTimerName(0, "CPU code");
    timer.setTimerName(1, "GPU kernel");

    int *in = NULL, *out = NULL, *out2 = NULL;

    cudaMallocHost(&in, sizeof(int)*ARRAY_SIZE);
    memset(in, 0, sizeof(int)*ARRAY_SIZE);

    cudaMallocHost(&out, sizeof(int)*ARRAY_SIZE);
    memset(out, 1, sizeof(int)*ARRAY_SIZE);

    cudaMallocHost(&out2, sizeof(int)*ARRAY_SIZE);
    memset(out2, 0, sizeof(int)*ARRAY_SIZE);

    LOOP_I(ARRAY_SIZE)
        in[i] = rand() % 10;

    // CPU-side equivalent computation for validation
    printf("CPU-side equivalent computation for validation...\n"); fflush(stdout);

    timer.onTimer(0);
#pragma omp parallel for num_threads(32)
    LOOP_I(ARRAY_SIZE) {
        int temp = 0;
        for (int j = 0; j < 250; j++) {
            temp = (temp + in[i] * 5) % 10;
        }
        out[i] = temp;
    }
    timer.offTimer(0);
    printf("Done!!\n");

    cudaStream_t stream[NUM_STREAMS];
    cudaEvent_t start[NUM_STREAMS], end[NUM_STREAMS];

    int* dIn[NUM_STREAMS];
    int* dOut[NUM_STREAMS];

    int ngpus;
    cudaGetDeviceCount(&ngpus);

    int chunkSize = ARRAY_SIZE / NUM_STREAMS;

    LOOP_I(NUM_STREAMS) {
        int devId = i % ngpus;
        cudaSetDevice(devId);

        cudaStreamCreate(&stream[i]);
        cudaEventCreate(&start[i]);
        cudaEventCreate(&end[i]);

        cudaMalloc(&dIn[i], sizeof(int) * chunkSize);
        cudaMalloc(&dOut[i], sizeof(int) * chunkSize);
    }

    timer.onTimer(1);

#pragma omp parallel for num_threads(NUM_STREAMS)
    LOOP_I(NUM_STREAMS)
    {
        int devId = i % ngpus;
        cudaSetDevice(devId);

        int offset = chunkSize * i;

        printf("Launching Stream[%d] on Device[%d]\n", i, devId);

        cudaEventRecord(start[i], stream[i]);

        cudaMemcpyAsync(dIn[i], in + offset, sizeof(int)*chunkSize, cudaMemcpyHostToDevice, stream[i]);
        myKernel2<<<NUM_BLOCK / NUM_STREAMS, NUM_T_IN_B, 0, stream[i]>>>(dIn[i], dOut[i]);
        cudaMemcpyAsync(out2 + offset, dOut[i], sizeof(int)*chunkSize, cudaMemcpyDeviceToHost, stream[i]);

        cudaEventRecord(end[i], stream[i]);

        //cudaStreamSynchronize(stream[i]); // Optional for accurate per-stream timing
        //cudaEventSynchronize(end[i]);
    }

    // Wait for all GPUs to finish
    LOOP_I(NUM_STREAMS) {
        cudaSetDevice(i % ngpus);
        cudaDeviceSynchronize();
    }

    timer.offTimer(1);
    timer.printTimer();

    float totalStreamTime = 0;
    LOOP_I(NUM_STREAMS) {
        float time = 0;
        cudaEventSynchronize(end[i]);
        cudaEventElapsedTime(&time, start[i], end[i]);
        totalStreamTime += time;
        printf("Stream[%d] : %f ms\n", i, time);
    }
    //printf("Sum of all stream GPU times = %f ms\n", totalStreamTime);

    bool mismatch_found = false;

    LOOP_I(ARRAY_SIZE) {
        if (out[i] != out2[i]) {
           printf("Data mismatch at index %d: expected %d, got %d\n", i, out[i], out2[i]);
           mismatch_found = true;
           break;
        }
    }

    if (!mismatch_found) {
        printf("Validation PASSED: All results match!\n");
    } else {
        printf("Validation FAILED: At least one mismatch found.\n");
    }


    LOOP_I(NUM_STREAMS) {
        cudaSetDevice(i % ngpus);
        cudaFree(dIn[i]);
        cudaFree(dOut[i]);

        cudaStreamDestroy(stream[i]);
        cudaEventDestroy(start[i]);
        cudaEventDestroy(end[i]);
    }

    cudaFreeHost(in);
    cudaFreeHost(out);
    cudaFreeHost(out2);

    return 0;
}
