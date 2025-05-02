# CUDA-Accelerated-Computing

A comprehensive collection of CUDA programming examples demonstrating GPU-accelerated computing, performance optimization, and parallel programming techniques. Many examples are adapted and refactored from tutorials and hands-on exercises developed during the [2025 Accelerator Programming Winter School](https://thunder.snu.ac.kr/aps) that I attended and the [Multi-core Programming course](https://www.youtube.com/playlist?list=PLBrGAFAIyf5pp3QNigbh2hRU5EUD0crgI) on YouTube.

## Overview

This repository provides hands-on examples that cover a wide range of CUDA programming concepts—from fundamental vector operations to advanced multi-GPU and multi-node computations. It’s designed to help developers understand and apply GPU acceleration in real-world applications.

## Project Structure

```
.
├── advanced_matmul/                # Advanced matrix multiplication implementations
├── convolution/                    # GPU-accelerated convolution operations
├── DSTimer/                        # Timing utilities for performance measurement
├── image_rotation/                 # Image rotation with OpenCV integration
├── image_rotation_without_OpenCV/  # OpenCV-independent image rotation
├── matmul/                         # Basic matrix multiplication examples
├── miscellaneous/                  # CUDA utilities and demonstrations
├── parallel_reduction/             # Parallel reduction patterns
├── pinned_memory/                  # Pinned memory examples
├── ptx/                            # PTX-related examples
├── vecadd/                         # Basic vector addition
└── vecsum_multinode/               # Multi-GPU and multi-node vector summation
```

## Examples Overview

### Basic Operations
- **Vector Addition**: Simple CUDA kernel for adding two vectors
- **Matrix Multiplication**: Various implementations including basic, tiled, and optimized versions

### Advanced Techniques
- **Tensor Core Operations**: Matrix multiplication using CUDA Tensor Cores
- **Shared Memory**: Optimization with CUDA shared memory
- **Stream Processing**: Overlapping computation and data transfer
- **Multi-GPU Programming**: Distributing work across multiple GPUs

### Image Processing
- **Image Rotation**: GPU-accelerated image rotation with and without OpenCV
- **Convolution**: 2D convolution implementations including im2col approach

### Performance Optimization
- **Parallel Reduction**: Efficient reduction patterns on GPU
- **Pinned Memory**: Optimizing host-device memory transfers
- **Stream & Event Management**: CUDA stream and event handling

## Requirements

- CMake 3.18+
- C++14 compatible compiler
- CUDA Toolkit 11.0+ (tested with CUDA 12.1)
- MPI implementation (e.g., OpenMPI 4.x) for multi-node examples
- OpenMP (optional for CPU parallel implementations)
- OpenCV 4.x (optional for image processing examples)

## Building the Project

### Environment Setup

If you're working in a HPC environment with environment modules, load the required modules:

```bash
# An example module configuration for KISTI GPU Cluster (Neuron: https://www.ksc.re.kr/eng/resources/neuron)
module load gcc/10.2.0 cuda/12.1 cudampi/openmpi-4.1.1 cmake/3.26.2

# Verify loaded modules
module list
# Expected output similar to:
# Currently Loaded Modules:
#   1) nvtop/1.1.0   3) singularity/4.1.0   5) cuda/12.1               7) cmake/3.26.2
#   2) htop/3.0.5    4) gcc/10.2.0          6) cudampi/openmpi-4.1.1
```

```bash
# Clone the repository
git clone https://github.com/hwang2006/CUDA-Accelerated-Computing.git
cd CUDA-Accelerated-Computing

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make -j8
```

### OpenCV Setup (Optional)

For image rotation examples with OpenCV:

1. Install OpenCV or build from source
2. Set OpenCV_DIR in CMake:
   ```
   set(OpenCV_DIR "/path/to/opencv/lib64/cmake/opencv4")
   ```

See `OpenCV_Build_Install_Guide.md` for detailed instructions.

## Running Examples

Each example can be run directly from the build directory:

```bash
# Run vector addition example
./vecadd/vecadd -h
Usage: ./vecadd/vecadd [-pvh]  [-n num_iterations] N
Options:
  -p : print vector data. (default: off)
  -v : validate vector addition. (default: off)
  -h : print this page.
  -t : number of threads (default: 1)
  -n : number of iterations (default: 1)
   N : number of elements of vectors. (default: 8)

# Run matrix multiplication
./matmul/matmul 
Options:
  Problem size: M = 8, N = 8, K = 8
  Number of iterations: 1
  Print matrix: off
  Validation: off

Initializing... done!
Calculating...(iter=0) 0.000189 sec
Avg. time: 0.000189 sec
Avg. throughput: 0.005423 GFLOPS

# Run image rotation (if OpenCV is available)
./image_rotation/image_rotation -h
Usage: ./image_rotation/image_rotation [-h] [-n num_src_images] [-d degree] [-i input_path]
Options:
  -h : print this page.
  -n : number of source images (default: 1)
  -d : rotation degree (default: 30)
  -i : input image path or directory (default: ./images)

# Run cuda stream example
./miscellaneous/cudaStream

*        DS_timer Report        *
* The number of timer = 10, counter = 10
**** Timer report ****
Single stream : 240.71800 ms (240.71800 ms)
  * Host -> Device : 21.19900 ms (21.19900 ms)
  * Kernel execution : 199.13400 ms (199.13400 ms)
  * Device -> Host : 20.38400 ms (20.38400 ms)
Multiple streams : 209.45700 ms (209.45700 ms)
**** Counter report ****
*        End of the report      *
 
```

## Performance Analysis

Most examples include timing measurements using the custom DSTimer utility. Results are printed to the console after execution.

## Learning Path

For those new to CUDA programming, we recommend following this sequence:

1. Start with `vecadd` to understand basic CUDA kernel execution
2. Move to `matmul` to learn about thread blocks and shared memory
3. Explore `image_rotation` for practical GPU image processing
4. Study `convolution` for more complex data manipulation
5. Examine `advanced_matmul` for optimization techniques
6. Investigate `vecsum_multinode` for multi-GPU programming

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions, feedback, or any other inquiries, please feel free to reach out:

- GitHub: [hwang2006](https://github.com/hwang2006)
- Email: hwang@kisti.re.kr
  
## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [2025 Accelerator Programming Winter School](https://thunder.snu.ac.kr/aps) for foundational CUDA examples and exercises
- [Multi-core Programming YouTube Course](https://www.youtube.com/playlist?list=PLBrGAFAIyf5pp3QNigbh2hRU5EUD0crgI) for inspiration and practical parallel computing techniques
- AI pair programming tools (e.g., ChatGPT, Gemini, Cluade, GitHub Copilot) for coding assistance and development support
- NVIDIA for CUDA documentation and examples
- The OpenCV community (for image processing examples)
