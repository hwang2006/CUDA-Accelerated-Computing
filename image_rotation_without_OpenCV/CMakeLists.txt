cmake_minimum_required(VERSION 3.18)
project(Image_rotation LANGUAGES CXX CUDA)

# Add the executable
add_executable(image_rotation_without_OpenCV
    main.cpp
    util.cpp
    image_rotation.cu
)

# Specify include directories using native CUDA support
target_include_directories(image_rotation_without_OpenCV PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA and OpenMP libraries using modern target-based approach
target_link_libraries(image_rotation_without_OpenCV PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

# Set CUDA compilation properties
set_target_properties(image_rotation_without_OpenCV PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
