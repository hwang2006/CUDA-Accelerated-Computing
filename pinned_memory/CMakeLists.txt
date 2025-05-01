cmake_minimum_required(VERSION 3.18)
project(Pinned_memory LANGUAGES CXX CUDA)

# Add the executable
add_executable(pinned_memory
    pinned_memory.cu 
)

# Specify include directories using native CUDA support
target_include_directories(pinned_memory PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(pinned_memory PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(pinned_memory PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(pinned_memory PRIVATE
    OpenMP::OpenMP_CXX
)


# Optional: Install the executable
install(TARGETS pinned_memory DESTINATION bin)
