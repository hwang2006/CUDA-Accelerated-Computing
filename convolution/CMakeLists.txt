cmake_minimum_required(VERSION 3.18)
project(Convolution LANGUAGES CXX CUDA)

#find_package(CUDAToolkit REQUIRED)

# Add the executable
add_executable(convolution
    main.cpp
    util.cpp
    convolution.cu
)

add_executable(convolution-v1
    main-v1.cpp
    util.cpp
    convolution-v1.cu
)

add_executable(conv2d_cpu
    conv2d_cpu.cpp
)

add_executable(convolution_im2col
    im2col/main.cpp
    im2col/util.cpp
    im2col/convolution_im2col.cu
)

add_executable(convolution_im2col_TensorCore
    im2col_TC/main.cpp
    im2col_TC/util.cpp
    im2col_TC/convolution_im2col_TC.cu
)

add_executable(conv2d_im2col_cpu
    im2col/conv2d_im2col.cpp
)

# Specify include directories using native CUDA support
target_include_directories(convolution PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(convolution PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(convolution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify include directories using native CUDA support
target_include_directories(convolution-v1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(convolution-v1 PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(convolution-v1 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify include directories using native CUDA support
target_include_directories(convolution_im2col PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/im2col
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(convolution_im2col PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(convolution_im2col PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify include directories using native CUDA support
target_include_directories(convolution_im2col_TensorCore PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/im2col
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(convolution_im2col_TensorCore PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(convolution_im2col_TensorCore PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(conv2d_cpu PRIVATE
    OpenMP::OpenMP_CXX
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(conv2d_im2col_cpu PRIVATE
    OpenMP::OpenMP_CXX
)

install(TARGETS convolution convolution-v1 convolution_im2col conv2d_cpu conv2d_im2col_cpu DESTINATION bin)
