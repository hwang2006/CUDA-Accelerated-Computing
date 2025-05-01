#include <mpi.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>  // for gethostname

#define THREADS_PER_BLOCK 1024
#define NUM_BLOCK (1024 * 10)
#define NUM_DATA (THREADS_PER_BLOCK * NUM_BLOCK)
#define NUM_STREAMS_PER_GPU 4

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void vecAdd(int *_a, int *_b, int *_c, int _size) {
    size_t tID = blockIdx.x * blockDim.x + threadIdx.x;
    if (tID < _size)
        _c[tID] = _a[tID] + _b[tID];
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    DS_timer timer(2);
    timer.setTimerName(0, "CUDA Total");
    timer.setTimerName(1, "VectorSum on Host");

    int numDevices;
    CHECK_CUDA(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        printf("No CUDA-capable GPU found on rank %d.\n", world_rank);
        MPI_Finalize();
        return -1;
    }

    size_t local_NUM_DATA = NUM_DATA / world_size;
    size_t memSize = sizeof(int) * local_NUM_DATA;
    if (world_rank == 0)
        printf("Total: %zu elements, per-rank: %zu, memSize per-rank = %zu bytes, using %d nodes * %d GPUs\n", NUM_DATA, local_NUM_DATA, memSize, world_size, numDevices);

    int *a, *b, *c;
    CHECK_CUDA(cudaMallocHost(&a, memSize));
    CHECK_CUDA(cudaMallocHost(&b, memSize));
    CHECK_CUDA(cudaMallocHost(&c, memSize));

    int *full_a = nullptr, *full_b = nullptr, *full_c = nullptr, *h_c = nullptr;
    if (world_rank == 0) {
        CHECK_CUDA(cudaMallocHost(&full_a, NUM_DATA * sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&full_b, NUM_DATA * sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&h_c, NUM_DATA * sizeof(int)));

        full_c = (int*)malloc(NUM_DATA * sizeof(int));

        srand(42);
        for (size_t i = 0; i < NUM_DATA; ++i) {
            full_a[i] = rand() % 10;
            full_b[i] = rand() % 10;
        }
    }

    MPI_Scatter(full_a, local_NUM_DATA, MPI_INT, a, local_NUM_DATA, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(full_b, local_NUM_DATA, MPI_INT, b, local_NUM_DATA, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        timer.onTimer(1);
        for (size_t i = 0; i < NUM_DATA; ++i)
            full_c[i] = full_a[i] + full_b[i];
        timer.offTimer(1);
    }

    size_t chunkPerGPU = local_NUM_DATA / numDevices;
    size_t chunkPerStream = chunkPerGPU / NUM_STREAMS_PER_GPU;
    size_t chunkBytes = chunkPerStream * sizeof(int);
    dim3 dimBlock(THREADS_PER_BLOCK);
    dim3 dimGrid((chunkPerStream + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    timer.onTimer(0);

    for (int dev = 0; dev < numDevices; dev++) {
        CHECK_CUDA(cudaSetDevice(dev));

        int *d_a, *d_b, *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, chunkPerGPU * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_b, chunkPerGPU * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_c, chunkPerGPU * sizeof(int)));

        cudaStream_t streams[NUM_STREAMS_PER_GPU];
        cudaEvent_t start[NUM_STREAMS_PER_GPU], end[NUM_STREAMS_PER_GPU];

        for (int s = 0; s < NUM_STREAMS_PER_GPU; s++) {
            CHECK_CUDA(cudaStreamCreate(&streams[s]));
            CHECK_CUDA(cudaEventCreate(&start[s]));
            CHECK_CUDA(cudaEventCreate(&end[s]));

            size_t local_in_offset = dev * chunkPerGPU + s * chunkPerStream;

            CHECK_CUDA(cudaEventRecord(start[s], streams[s]));
            CHECK_CUDA(cudaMemcpyAsync(d_a + s * chunkPerStream, a + local_in_offset, chunkBytes, cudaMemcpyHostToDevice, streams[s]));
            CHECK_CUDA(cudaMemcpyAsync(d_b + s * chunkPerStream, b + local_in_offset, chunkBytes, cudaMemcpyHostToDevice, streams[s]));

            vecAdd<<<dimGrid, dimBlock, 0, streams[s]>>>(
                d_a + s * chunkPerStream,
                d_b + s * chunkPerStream,
                d_c + s * chunkPerStream,
                chunkPerStream);
            CHECK_CUDA(cudaGetLastError());

            CHECK_CUDA(cudaMemcpyAsync(c + local_in_offset, d_c + s * chunkPerStream, chunkBytes, cudaMemcpyDeviceToHost, streams[s]));
            CHECK_CUDA(cudaEventRecord(end[s], streams[s]));
        }

        for (int s = 0; s < NUM_STREAMS_PER_GPU; s++) {
            CHECK_CUDA(cudaStreamSynchronize(streams[s]));
            float time = 0;
            CHECK_CUDA(cudaEventElapsedTime(&time, start[s], end[s]));
            printf("Rank[%d] Host[%s] Device[%d] Stream[%d]: %.2f ms\n", world_rank, hostname, dev, s, time);
            fflush(stdout);
            CHECK_CUDA(cudaEventDestroy(start[s]));
            CHECK_CUDA(cudaEventDestroy(end[s]));
            CHECK_CUDA(cudaStreamDestroy(streams[s]));
        }

        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }

    timer.offTimer(0);
    timer.printTimer();

    if (world_rank == 0) {
        //MPI_Gather(MPI_IN_PLACE, local_NUM_DATA, MPI_INT, h_c, local_NUM_DATA, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(c, local_NUM_DATA, MPI_INT, h_c, local_NUM_DATA, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(c, local_NUM_DATA, MPI_INT, nullptr, local_NUM_DATA, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (world_rank == 0) {
        bool correct = true;
        for (size_t i = 0; i < NUM_DATA; i++) {
            if (h_c[i] != full_c[i]) {
                //printf("Mismatch at %zu\n", i);
                printf("Mismatch at %zu: GPU=%d, CPU=%d\n", i, h_c[i], full_c[i]);
                correct = false;
                break;
            }
        }
        if (correct) {
            printf("GPU computation is correct across %d nodes and %d GPUs per node!\n", world_size, numDevices);
            fflush(stdout);
        }
    }

    CHECK_CUDA(cudaFreeHost(a));
    CHECK_CUDA(cudaFreeHost(b));
    CHECK_CUDA(cudaFreeHost(c));
    if (world_rank == 0) {
        CHECK_CUDA(cudaFreeHost(full_a));
        CHECK_CUDA(cudaFreeHost(full_b));
        CHECK_CUDA(cudaFreeHost(h_c));
        free(full_c);
    }

    MPI_Finalize();
    return 0;
}
