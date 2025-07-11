/**
This is an exmple code used in the CUDA Lecture 4 (Quick Lab. 11-2) <br>
@author : Duksu Kim
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <DS_timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOOP_I(_loop) for(int i=0; i < _loop; i++)

#define NUM_BLOCK (128*1024)
#define NUM_T_IN_B 1024
#define ARRAY_SIZE (NUM_T_IN_B*NUM_BLOCK)

//#define NUM_STREAMS 2
#define NUM_STREAMS 4

__global__ void myKernel(int *_in, int *_out)
{
	int tID = blockDim.x * blockIdx.x + threadIdx.x;

	int temp = 0;
	//for (int i = 0; i < 250; i++) {
	for (int i = 0; i < 500; i++) {
		temp = (temp + _in[tID] * 5) % 10;
	}
	_out[tID] = temp;

}

//void main_stream(void)
int main(void)
{
	DS_timer timer(10);
	timer.setTimerName(0, "Single stream");
	timer.setTimerName(1, "  * Host -> Device");
	timer.setTimerName(2, "  * Kernel execution");
	timer.setTimerName(3, "  * Device -> Host");
	timer.setTimerName(4, "Multiple streams");

	int *in = NULL, *out = NULL, *out2 = NULL;

	cudaMallocHost(&in, sizeof(int)*ARRAY_SIZE);
	memset(in, 0, sizeof(int)*ARRAY_SIZE);

	cudaMallocHost(&out, sizeof(int)*ARRAY_SIZE);
	memset(out, 0, sizeof(int)*ARRAY_SIZE);

	cudaMallocHost(&out2, sizeof(int)*ARRAY_SIZE);
	memset(out2, 0, sizeof(int)*ARRAY_SIZE);

	int *dIn, *dOut;
	cudaMalloc(&dIn, sizeof(int)*ARRAY_SIZE);
	cudaMalloc(&dOut, sizeof(int)*ARRAY_SIZE);

	LOOP_I(ARRAY_SIZE)
		in[i] = rand() % 10;

	// Single stram version
	timer.onTimer(0);

	timer.onTimer(1);
	cudaMemcpy(dIn, in, sizeof(int)*ARRAY_SIZE, cudaMemcpyHostToDevice);
	timer.offTimer(1);

	timer.onTimer(2);
	myKernel << <NUM_BLOCK, NUM_T_IN_B >> > (dIn, dOut);
	cudaDeviceSynchronize();
	timer.offTimer(2);

	timer.onTimer(3);
	cudaMemcpy(out, dOut, sizeof(int)*ARRAY_SIZE, cudaMemcpyDeviceToHost);
	timer.offTimer(3);

	timer.offTimer(0);

	// Multiple stream version
	cudaStream_t stream[NUM_STREAMS];

	LOOP_I(NUM_STREAMS)
		cudaStreamCreate(&stream[i]);

	timer.onTimer(4);

	int chunkSize = ARRAY_SIZE / NUM_STREAMS;

	LOOP_I(NUM_STREAMS)
	{
		int offset = chunkSize * i;
		cudaMemcpyAsync(dIn + offset, in + offset, sizeof(int)*chunkSize, cudaMemcpyHostToDevice, stream[i]);
		myKernel << <NUM_BLOCK / NUM_STREAMS, NUM_T_IN_B, 0, stream[i] >> > (dIn + offset, dOut + offset);
		cudaMemcpyAsync(out2 + offset, dOut + offset, sizeof(int)*chunkSize, cudaMemcpyDeviceToHost, stream[i]);
	}

	cudaDeviceSynchronize();
	timer.offTimer(4);

	LOOP_I(ARRAY_SIZE) {
		if (out[i] != out2[i])
			printf("!");
	}

	LOOP_I(NUM_STREAMS)
		cudaStreamDestroy(stream[i]);

	timer.printTimer();

	cudaFree(dIn);
	cudaFree(dOut);

	cudaFreeHost(in);
	cudaFreeHost(out);
	cudaFreeHost(out2);

    return 0; 
}
