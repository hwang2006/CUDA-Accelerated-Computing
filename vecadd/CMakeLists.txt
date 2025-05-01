cmake_minimum_required(VERSION 3.18)
project(MatMul LANGUAGES CXX CUDA)

# Add the executable
add_executable(vecadd
    main.cpp
    util.cpp
    vecadd.cu
)

# Specify include directories using native CUDA support
target_include_directories(vecadd PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(vecadd PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(vecadd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optional: Install the executable
install(TARGETS vecadd DESTINATION bin)
