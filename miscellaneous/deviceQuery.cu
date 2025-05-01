/**
This is an exmple code used in the CUDA Lecture 5 (Quick Lab. 12-1) <br>
@author : Duksu Kim
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
	int ngpus;
	cudaGetDeviceCount(&ngpus);

	for (int i = 0; i < ngpus; i++) {
		cudaDeviceProp devProp;

		cudaGetDeviceProperties(&devProp, i);
		printf("Device[%d](%s) compute capability : %d.%d.\n"
			, i, devProp.name, devProp.major, devProp.minor);
	}
}
