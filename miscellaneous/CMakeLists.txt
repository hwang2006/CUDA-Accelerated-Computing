cmake_minimum_required(VERSION 3.18)
project(MatMul LANGUAGES CXX CUDA)

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
add_executable(cudaStream
    cudaStream.cu 
    ../DSTimer/DS_timer.cpp
)
set_cuda_target(cudaStream)

add_executable(cudaEvent
    cudaEvent.cu
    ../DSTimer/DS_timer.cpp
)
set_cuda_target(cudaEvent)


add_executable(threadCounting
    threadCounting.cu
    ../DSTimer/DS_timer.cpp
)
set_cuda_target(threadCounting)


add_executable(deviceQuery
    deviceQuery.cu
    ../DSTimer/DS_timer.cpp
)
set_cuda_target(deviceQuery)


add_executable(cudaEvent_multiGPUs
    cudaEvent_multiGPUs.cu
    ../DSTimer/DS_timer.cpp
)
set_cuda_target(cudaEvent_multiGPUs)

# Specify include directories using native CUDA support
#target_include_directories(cudaStream PRIVATE
#    ${CMAKE_CURRENT_SOURCE_DIR}
#    ${CMAKE_CURRENT_SOURCE_DIR}/../DSTimer
#    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#)

# Link CUDA and OpenMP libraries using modern target-based approach
#target_link_libraries(cudaStream PRIVATE
#    cudart
#    OpenMP::OpenMP_CXX
#)

# Set CUDA compilation properties
#set_target_properties(cudaStream PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON
#)

# Optional: Install the executable
install(TARGETS cudaStream cudaEvent threadCounting deviceQuery DESTINATION bin)
