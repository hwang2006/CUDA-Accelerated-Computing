cmake_minimum_required(VERSION 3.18)
project(MatMul LANGUAGES CXX CUDA)

# Add the executable
add_executable(cpu_matmul
    cpu_matmul.cpp
)

add_executable(matmul
    main.cpp
    util.cpp
    matmul.cu
)

add_executable(matmul_basic
    main.cpp
    util.cpp
    matmul_basic.cu
)

add_executable(matmul_unrolling
    main.cpp
    util.cpp
    matmul_unrolling.cu
)

add_executable(matmul_unrolling_stream
    main.cpp
    util.cpp
    matmul_unrolling_stream.cu
)

add_executable(matmul_stream
    main.cpp
    util.cpp
    matmul_stream.cu
)

add_executable(matmul_comparison
    matmul_comparision.cu
    ../DSTimer/DS_timer.cpp
)


# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(cpu_matmul PRIVATE
    OpenMP::OpenMP_CXX
)


# Specify include directories using native CUDA support
target_include_directories(matmul PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(matmul PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify include directories using native CUDA support
target_include_directories(matmul_basic PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(matmul_basic PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(matmul_basic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


# Specify include directories using native CUDA support
target_include_directories(matmul_unrolling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(matmul_unrolling PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(matmul_unrolling PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify include directories using native CUDA support
target_include_directories(matmul_unrolling_stream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(matmul_unrolling_stream PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(matmul_unrolling_stream PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


# Specify include directories using native CUDA support
target_include_directories(matmul_stream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(matmul_stream PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(matmul_stream PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Set CUDA compilation properties
set_target_properties(matmul_stream PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify include directories using native CUDA support
target_include_directories(matmul_comparison PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../DSTimer
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(matmul_comparison PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Optional: Install the executable
install(TARGETS matmul matmul_basic matmul_unrolling matmul_unrolling_stream matmul_stream matmul_comparison DESTINATION bin)
