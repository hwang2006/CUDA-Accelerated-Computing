cmake_minimum_required(VERSION 3.18)
project(MatMul LANGUAGES CXX CUDA)

# Add the executable
add_executable(matmul_shared
    main.cpp
    util.cpp
    matmul_shared.cu
)

add_executable(matmul_shared_singleBlock 
    matmul_shared_singleBlock.cu 
    ../DSTimer/DS_timer.cpp
)

add_executable(matmul_fp16_TensorCore
    matmul_fp16_TC/main.cpp
    matmul_fp16_TC/util.cpp
    matmul_fp16_TC/matmul_fp16_TC.cu
    matmul_fp16_TC/matmul_fp16_standard.cu
)

add_executable(matmul_TC_singleBlock
    matmul_fp16_TC/matmul_TC_singleBlock.cu
    ../DSTimer/DS_timer.cpp
)

add_executable(test_TC
    matmul_fp16_TC/test_TC.cu
    ../DSTimer/DS_timer.cpp
)


# Specify include directories using native CUDA support
target_include_directories(matmul_shared PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(matmul_shared PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(matmul_shared PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify include directories using native CUDA support
target_include_directories(matmul_shared_singleBlock PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_fp16_TC/
    ${CMAKE_CURRENT_SOURCE_DIR}/../DSTimer
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(matmul_shared_singleBlock PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(matmul_shared_singleBlock PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify include directories using native CUDA support
target_include_directories(matmul_fp16_TensorCore PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_fp16_TC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(matmul_fp16_TensorCore PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(matmul_fp16_TensorCore PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify include directories using native CUDA support
target_include_directories(matmul_TC_singleBlock PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_fp16_TC
    ${CMAKE_CURRENT_SOURCE_DIR}/../DSTimer
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(matmul_TC_singleBlock PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(matmul_TC_singleBlock PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify include directories using native CUDA support
target_include_directories(test_TC PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_fp16_TC
    ${CMAKE_CURRENT_SOURCE_DIR}/../DSTimer
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(test_TC PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(test_TC PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)



# Optional: Install the executable
install(TARGETS matmul_shared matmul_shared_singleBlock matmul_fp16_TensorCore matmul_TC_singleBlock test_TC DESTINATION bin)
