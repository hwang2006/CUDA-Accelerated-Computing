cmake_minimum_required(VERSION 3.18)
project(VecSum LANGUAGES CXX CUDA)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)

# MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# Add CUDA include directories globally
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# ---------------------------
# CUDA Target Helper Function
# ---------------------------

# Simplify CUDA target settings
function(set_cuda_target target)
    target_include_directories(${target} PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../DSTimer
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_compile_features(${target} PRIVATE cuda_std_14)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

# Add the executable
add_executable(vecSum
    vecSum.cu
    ../DSTimer/DS_timer.cpp
)
set_cuda_target(vecSum)

add_executable(vecSum_stream
    vecSum_stream.cu
    ../DSTimer/DS_timer.cpp
)
set_cuda_target(vecSum_stream)


add_executable(vecSum_stream_multiGPUs
    vecSum_stream_multiGPUs.cu
    ../DSTimer/DS_timer.cpp
)
set_cuda_target(vecSum_stream_multiGPUs)


add_executable(vecSum_multiNodes
    vecSum_multiNodes.cu
    ../DSTimer/DS_timer.cpp
)
set_cuda_target(vecSum_multiNodes)
target_link_libraries(vecSum_multiNodes PRIVATE ${MPI_CXX_LIBRARIES})


# Optional: Install the executable
install(TARGETS vecSum vecSum_stream vecSum_stream_multiGPUs vecSum_multiNodes DESTINATION bin)
