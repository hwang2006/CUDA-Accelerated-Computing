/**
This is an exmple solution code for Lab 4-1. Vector sum for large vectors <br>
@author : Duksu Kim
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The size of the vector
#define NUM_DATA (1024*1024*256)
//#define NUM_DATA ((size_t)(1024 * 1024 * 1024))

__global__ void vecAdd(int *_a, int *_b, int *_c, int _size) {
	int tID = blockIdx.x * blockDim.x + threadIdx.x;
	if (tID < _size)
		_c[tID] = _a[tID] + _b[tID];
}

int main(void)
{
	// Set timer
	DS_timer timer(5);
	timer.setTimerName(0, "CUDA Total");
	timer.setTimerName(1, "Computation(Kernel)");
	timer.setTimerName(2, "Data Trans. : Host -> Device");
	timer.setTimerName(3, "Data Trans. : Device -> Host");
	timer.setTimerName(4, "VectorSum on Host");
	timer.initTimers();
	//timer.timerOff();

	int *a, *b, *c, *h_c;	// Vectors on the host
	int *d_a, *d_b, *d_c;	// Vectors on the device

	//int memSize = sizeof(int)*NUM_DATA;
	size_t memSize = sizeof(int)*NUM_DATA;  //g long 
	//printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);
	printf("%zu elements, memSize = %zu bytes\n", NUM_DATA, memSize);

	// Memory allocation on the host-side
	//a = new int[NUM_DATA]; memset(a, 0, memSize);
	cudaMallocHost(&a, memSize); memset(a, 0, memSize);  //pinned memory
	//b = new int[NUM_DATA]; memset(b, 0, memSize);
	cudaMallocHost(&b, memSize); memset(b, 0, memSize);
	//c = new int[NUM_DATA]; memset(c, 0, memSize);
	cudaMallocHost(&c, memSize); memset(c, 0, memSize);
	//h_c = new int[NUM_DATA]; memset(h_c, 0, memSize);
    h_c = (int *)malloc(memSize); memset(h_c, 0, memSize);

	// Data generation
	for (int i = 0; i < NUM_DATA; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	// Vector sum on host (for performance comparision)
	timer.onTimer(4);
	for (int i = 0; i < NUM_DATA; i++)
		h_c[i] = a[i] + b[i];
	timer.offTimer(4);

	// Memory allocation on the device-side
	cudaMalloc(&d_a, memSize);
	cudaMalloc(&d_b, memSize);
	cudaMalloc(&d_c, memSize);

	timer.onTimer(0);

	// Data copy : Host -> Device
	timer.onTimer(2);
	cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);
	timer.offTimer(2);

	// Kernel call
	//dim3 dimGrid(ceil((float)NUM_DATA / 256), 1, 1);
	dim3 dimGrid(ceil((float)NUM_DATA / 1024), 1, 1);
	//dim3 dimBlock(256, 1, 1);
	dim3 dimBlock(1024, 1, 1);

	timer.onTimer(1);
	vecAdd <<<dimGrid, dimBlock>>>(d_a, d_b, d_c, NUM_DATA);
	cudaDeviceSynchronize(); // synchronization function
	timer.offTimer(1);

	// Copy results : Device -> Host
	timer.onTimer(3);
	cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
	timer.offTimer(3);

	timer.offTimer(0); timer.printTimer();

	// Check results
	bool result = true;
	for (int i = 0; i < NUM_DATA; i++) {
		if (h_c[i] != c[i]) {
			printf("[%d] The resutls is not matched! (%d, %d)\n"
				, i, h_c[i], c[i]);
			result = false;
		}
	}

	if (result)
		printf("GPU works well!\n");

	// Release device memory
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	// Release host memory
	//delete[] a; delete[] b; delete[] c;
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);

	return 0;
}
