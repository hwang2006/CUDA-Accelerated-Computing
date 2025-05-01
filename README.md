# CUDA-Accelerated-Computing

A comprehensive collection of CUDA programming examples demonstrating GPU-accelerated computing, performance optimization, and parallel programming techniques.

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
# An example module configuration
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
./vecadd/vecadd

# Run matrix multiplication
./matmul/matmul

# Run image rotation (if OpenCV is available)
./image_rotation/image_rotation
```

## Performance Benchmarking

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

- NVIDIA for CUDA documentation and examples
- The OpenCV community (for image processing examples)


# CUDA Accelerated Computing Examples

A comprehensive collection of CUDA programming examples demonstrating GPU-accelerated computing concepts, optimization techniques, and parallel programming patterns.

## Overview

This repository contains practical implementations of various GPU computing paradigms using CUDA. From basic vector operations to advanced tensor computations, these examples provide hands-on experience with CUDA programming and optimization strategies.

## Project Structure

```
.
├── advanced_matmul/       # Advanced matrix multiplication implementations
├── convolution/           # GPU-accelerated convolution operations
├── DSTimer/               # Timing utilities for performance measurement
├── image_rotation/        # Image rotation with OpenCV integration
├── image_rotation_without_OpenCV/ # OpenCV-independent image rotation
├── matmul/                # Basic matrix multiplication examples
├── miscellaneous/         # CUDA utilities and demonstrations
├── parallel_reduction/    # Parallel reduction patterns
├── pinned_memory/         # Pinned memory examples
├── ptx/                   # PTX-related examples
├── vecadd/                # Basic vector addition
└── vecsum_multinode/      # Multi-GPU and multi-node vector summation
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

- CUDA Toolkit 11.0+ (tested with CUDA 12.1)
- CMake 3.18+
- C++14 compatible compiler
- MPI implementation (e.g., OpenMPI 4.x) for multi-node examples
- OpenMP (optional for CPU parallel implementations)
- OpenCV 4.x (optional for image processing examples)

## Building the Project

### Environment Setup

If you're working in an HPC environment with environment modules, load the required modules:

```bash
# Example module configuration
module load gcc/10.2.0 cuda/12.1 cudampi/openmpi-4.1.1 cmake/3.26.2

# Verify loaded modules
module list
# Expected output similar to:
# Currently Loaded Modules:
#   1) nvtop/1.1.0   3) singularity/4.1.0   5) cuda/12.1               7) cmake/3.26.2
#   2) htop/3.0.5    4) gcc/10.2.0          6) cudampi/openmpi-4.1.1
```

### Compilation

```bash
# Clone the repository
git clone https://github.com/yourusername/cuda-accelerated-computing-examples.git
cd cuda-accelerated-computing-examples

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
   set(OpenCV_DIR "/path/to/opencv/lib/cmake/opencv4")
   ```

See `OpenCV_Build_Install_Guide.md` for detailed instructions.

## Running Examples

Each example can be run directly from the build directory:

```bash
# Run vector addition example
./vecadd/vecadd

# Run matrix multiplication
./matmul/matmul

# Run image rotation (if OpenCV is available)
./image_rotation/image_rotation
```

## Performance Benchmarking

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA for CUDA documentation and examples
- The OpenCV community (for image processing examples)

## Contact

For questions, feedback, or collaboration:

**Soonwook Hwang**  
Email: hwang@kisti.re.kr  
Korea Institute of Science and Technology Information (KISTI)
