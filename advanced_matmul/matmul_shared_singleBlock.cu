/**
This is an exmple code used in the CUDA Lecture 2 (Quick Lab. 9-2) <br>
@author : Duksu Kim
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <DS_timer.h>

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_CPU_THREADS	(32)

#define ROW_SIZE (32)
//#define ROW_SIZE (16)
#define K_SIZE   (128)
//#define K_SIZE   (1024)
#define COL_SIZE (32)
//#define COL_SIZE (64)

#if K_SIZE < 129
#define USE_SHARED_VER
#endif
#define USE_SHARED_VER_C
#define USE_BASE_KERNEL

#define	WORK_LOAD (1024)
#define MAT_SIZE_A (ROW_SIZE*K_SIZE)
#define MAT_SIZE_B (K_SIZE*COL_SIZE)
#define MAT_SIZE_C (ROW_SIZE*COL_SIZE)

// input matrix
float A[ROW_SIZE][K_SIZE];	// m * k
float B[K_SIZE][COL_SIZE];	// k * n

// timer
DS_timer* timer;
void setTimer(void);
#define TIMER_HOST		0
#define TIMER_KERNEL	1
#define TIMER_KERNEL_xRow	2
#define TIMER_KERNEL_SH	3
#define TIMER_KERNEL_SH_C	4
#define TIMER_HtoD		5
#define TIMER_DtoH		6
#define NUM_TIMER		(TIMER_DtoH+1)

void genInputMatrices(void);

void checkResult(const char *);

// output matrix
float hostC[ROW_SIZE][COL_SIZE];	// host result
float deviceC[ROW_SIZE][COL_SIZE];	// device result

#define memsetZero(_P,_type,_size) memset(_P, 0, sizeof(_type)*_size);
#define dMemAlloc(_P, _type, _size) cudaMalloc(&_P, sizeof(_type)*_size);

#ifdef USE_BASE_KERNEL
__global__ void matMul_kernel(float* _A, float* _B, float* _C)
{
    //dim3 blockDim(COL_SIZE, ROW_SIZE)
	int row = threadIdx.y;  
	int col = threadIdx.x;
	int index = row * blockDim.x + col;

	_C[index] = 0;
	for (int k = 0 ; k < K_SIZE ; k++)
		for (int i = 0 ; i < WORK_LOAD; i++)
			_C[index] += _A[row * K_SIZE + k] * _B[col + k * COL_SIZE];
}

__global__ void matMul_kernel_xRow(float* _A, float* _B, float* _C)
{
    //dim3 blockDim(ROW_SIZE, COL_SIZE)
	int row = threadIdx.x;
	int col = threadIdx.y;
	int index = row * blockDim.y + col;

	_C[index] = 0;
	for (int k = 0; k < K_SIZE; k++)
		for (int i = 0; i < WORK_LOAD; i++)
			_C[index] += _A[row * K_SIZE + k] * _B[col + k * COL_SIZE];
}

#endif

#ifdef USE_SHARED_VER
__global__ void matMul_kernel_shared(float* _A, float* _B, float* _C)
{
	int row = threadIdx.y;
	int col = threadIdx.x;
	int index = row * blockDim.x + col;

	__shared__ float sA[ROW_SIZE][K_SIZE];	// 32 * 256 * 4 bytes = 16 KB
	__shared__ float sB[K_SIZE][COL_SIZE];	// 16 KB

	int offset = 0;

	// load A
	int numSubMatA = ceil((float)K_SIZE / COL_SIZE );
	for (int i = 0; i < numSubMatA; i++) {
		if (col + offset >= K_SIZE) break;

		sA[row][col + offset] = _A[row * K_SIZE + (col + offset)];
		offset += COL_SIZE;
	}

	// load B
	offset = 0;
	int numSubMatB = ceil((float) K_SIZE / ROW_SIZE);
	for (int i = 0; i < numSubMatB; i++) {
		if (row + offset >= K_SIZE) break;

		sB[row + offset][col] = _B[col + (row + offset) * COL_SIZE];
		offset += ROW_SIZE;
	}

	__syncthreads(); // wait until all thread load the matrix

	_C[index] = 0;
	for (int k = 0; k < K_SIZE; k++)
		for (int i = 0; i < WORK_LOAD; i++)
			_C[index] += sA[row][k] * sB[k][col];
}
#endif

#ifdef USE_SHARED_VER_C
__global__ void matMul_kernel_shared_C(float* _A, float* _B, float* _C)
{
	int row = threadIdx.y;
	int col = threadIdx.x;
	int index = row * blockDim.x + col;

	__shared__ float sC[MAT_SIZE_C];

	sC[index] = 0;
	for (int k = 0; k < K_SIZE; k++)
		for (int i = 0; i < WORK_LOAD; i++)
			sC[index] += _A[row * K_SIZE + k] * _B[col + k * COL_SIZE];

	_C[index] = sC[index];
}
#endif

//void main(void)
int main(void)
{
	timer = NULL;	setTimer();

	float *dA, *dB, *dC;
	dA = dB = dC = NULL;

	memsetZero(A, float, MAT_SIZE_A);	memsetZero(B, float, MAT_SIZE_B);
	memsetZero(hostC, float, MAT_SIZE_C);	memsetZero(deviceC, float, MAT_SIZE_C);

	// device memory allocaiton
	dMemAlloc(dA, float, MAT_SIZE_A);
	dMemAlloc(dB, float, MAT_SIZE_B);
	dMemAlloc(dC, float, MAT_SIZE_C);

	// generate input matrices
	genInputMatrices();

	// Host code
	timer->onTimer(TIMER_HOST);
	//#pragma omp parallel for num_threads(NUM_CPU_THREADS)
	for (int r = 0 ; r < ROW_SIZE ; r++)
		for (int c = 0; c < COL_SIZE; c++)
			for (int k = 0 ; k < K_SIZE; k++)
				for (int i = 0 ; i < WORK_LOAD; i++)
					hostC[r][c] += A[r][k] * B[k][c];
	timer->offTimer(TIMER_HOST);

	// Copy input matrices : H -> D
	timer->onTimer(TIMER_HtoD);
	cudaMemcpy(dA, A, sizeof(float)*MAT_SIZE_A, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeof(float)*MAT_SIZE_B, cudaMemcpyHostToDevice);
	timer->offTimer(TIMER_HtoD);

	dim3 blockDim(COL_SIZE, ROW_SIZE); //blockDim.x=COL_SIZE, blockDim.y=ROW_SIZE

#ifdef USE_BASE_KERNEL
	//// Kernel call
	timer->onTimer(TIMER_KERNEL);
	
	matMul_kernel <<<1,  blockDim>>> (dA, dB, dC);
	cudaDeviceSynchronize();
    timer->offTimer(TIMER_KERNEL);
    cudaMemcpy(deviceC, dC, sizeof(float)*MAT_SIZE_C, cudaMemcpyDeviceToHost);
    checkResult("KERNEL");
    memsetZero(deviceC, float, MAT_SIZE_C);
        

	// For matMul_kernel_xRow
	blockDim.x = ROW_SIZE;
	blockDim.y = COL_SIZE;
    timer->onTimer(TIMER_KERNEL_xRow);
	matMul_kernel_xRow <<<1, blockDim >> > (dA, dB, dC);
	cudaDeviceSynchronize();
	timer->offTimer(TIMER_KERNEL_xRow);
    cudaMemcpy(deviceC, dC, sizeof(float)*MAT_SIZE_C, cudaMemcpyDeviceToHost);
    checkResult("KERNEL_xRow");
    memsetZero(deviceC, float, MAT_SIZE_C);
#endif

#ifdef USE_SHARED_VER
	//// Kernel call (shared memory)
    blockDim.x = COL_SIZE;
    blockDim.y = ROW_SIZE;
	timer->onTimer(TIMER_KERNEL_SH);
	matMul_kernel_shared << <1, blockDim >> > (dA, dB, dC);
	cudaDeviceSynchronize();
	timer->offTimer(TIMER_KERNEL_SH);
    cudaMemcpy(deviceC, dC, sizeof(float)*MAT_SIZE_C, cudaMemcpyDeviceToHost);
    checkResult("KERNEL_SH");
    memsetZero(deviceC, float, MAT_SIZE_C);
#endif

#ifdef USE_SHARED_VER_C
	//// Kernel call (shared memory C)
    blockDim.x = COL_SIZE;
    blockDim.y = ROW_SIZE;
	timer->onTimer(TIMER_KERNEL_SH_C);
	matMul_kernel_shared_C << <1, blockDim >> > (dA, dB, dC);
	cudaDeviceSynchronize();
	timer->offTimer(TIMER_KERNEL_SH_C);
    cudaMemcpy(deviceC, dC, sizeof(float)*MAT_SIZE_C, cudaMemcpyDeviceToHost);
    checkResult("KERNEL_SH_C");
    memsetZero(deviceC, float, MAT_SIZE_C);
#endif

	// Get back result : D -> H
	timer->onTimer(TIMER_DtoH);
	cudaMemcpy(deviceC, dC, sizeof(float)*MAT_SIZE_C, cudaMemcpyDeviceToHost);
	//timer->onTimer(TIMER_DtoH);
	timer->offTimer(TIMER_DtoH);

	// check the results
	bool isCorrect = true;
	
	float *pHostC = &hostC[0][0];
	float *pDeviceC = &deviceC[0][0];
	
	for (int i = 0; i < MAT_SIZE_C; i++){
		if (pHostC[i] != pDeviceC[i]) {
			printf("[%d] %.2f, %.2f\n", i, pHostC[i], pDeviceC[i]);
			isCorrect = false;
			break;
		}
	}

	if (isCorrect) printf("Result is correct!\n");
	else printf("Result is not correct!!!!!!\n");

	timer->printTimer();
	if (timer != NULL)
		delete timer;
    
    return 0;
}

void genInputMatrices(void)
{
	for (int r = 0; r < ROW_SIZE; r++)
		for (int k = 0; k < K_SIZE; k++)
			A[r][k] = rand() % 100;

	for (int k = 0; k < K_SIZE; k++)
		for (int c = 0; c < COL_SIZE; c++)
			B[k][c] = rand() % 100;

		// currently disable it for making a simple example while avoiding precision problem
		// A[r][c] += (float)(rand() % 100) / 100.0;
		// B[r][c] += (float)(rand() % 100) / 100.0;
}

void setTimer(void)
{
	timer = new DS_timer(NUM_TIMER);

	timer->initTimers();
	timer->setTimerName(TIMER_HOST, "CPU code");
	timer->setTimerName(TIMER_KERNEL, "Kernel launch");
	timer->setTimerName(TIMER_KERNEL_xRow, "Kernel launch with xRow");
	timer->setTimerName(TIMER_KERNEL_SH, "Kernel launch (shared ver.)");
	timer->setTimerName(TIMER_KERNEL_SH_C, "Kernel launch (shared ver. C)");
	timer->setTimerName(TIMER_HtoD, "[Data transter] host->device");
	timer->setTimerName(TIMER_DtoH, "[Data transfer] device->host");
}

void checkResult(const char *kernel_type)
{
    // check the results
    bool isCorrect = true;

    float *pHostC = &hostC[0][0];
    float *pDeviceC = &deviceC[0][0];

    for (int i = 0; i < MAT_SIZE_C; i++){
        if (pHostC[i] != pDeviceC[i]) {
            printf("[%d] %.2f, %.2f\n", i, pHostC[i], pDeviceC[i]);
            isCorrect = false;
            break;
        }
    }

    if (isCorrect) printf("%s: Result is correct!\n", kernel_type);
    else printf("%s: Result is not correct!!!!!!\n", kernel_type);
}


