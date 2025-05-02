cmake_minimum_required(VERSION 3.18)
project(CUDAProjects VERSION 1.0 LANGUAGES CXX CUDA)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Specify CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 90)

# Add CUDA include directories globally using native CUDA support
include_directories()

# Add this line before find_package(OpenCV REQUIRED)
set(OpenCV_DIR "/scratch/qualis/.local/opencv/lib64/cmake/opencv4")

# Check if OpenCV is installed
find_package(OpenCV)

# If OpenCV is found, add the image_rotation subdirectory
if(OpenCV_FOUND)
    message(STATUS "OpenCV found, adding image_rotation subdirectory.")
    add_subdirectory(image_rotation)
else()
    message(WARNING "OpenCV not found, skipping image_rotation subdirectory.")
endif()

# Add other subdirectories
add_subdirectory(vecadd)
add_subdirectory(vecsum_multinode)
add_subdirectory(matmul)
add_subdirectory(advanced_matmul)
add_subdirectory(image_rotation_without_OpenCV)
add_subdirectory(convolution)
add_subdirectory(pinned_memory)
add_subdirectory(parallel_reduction)
add_subdirectory(miscellaneous/)

