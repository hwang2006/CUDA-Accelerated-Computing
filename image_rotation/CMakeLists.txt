cmake_minimum_required(VERSION 3.18)
project(Image_rotation LANGUAGES CXX CUDA)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(image_rotation_gray
    main_gray.cpp
    util.cpp
    image_rotation.cu
)

add_executable(image_rotation
    main.cpp
    util.cpp
    image_rotation.cu
)

add_executable(rotate_image_cpu_v1
    cpp-version/rotate_image_v1.cpp
    cpp-version/util.cpp
)

add_executable(rotate_image_cpu_v2
    cpp-version/rotate_image_v2.cpp
    cpp-version/util.cpp
)

add_executable(rotate_image_cpu_v3
    cpp-version/rotate_image_v3.cpp
    cpp-version/util.cpp
)

add_executable(rotate_image_cpu_v4
    cpp-version/rotate_image_v4.cpp
    cpp-version/util.cpp
)

add_executable(rotate_image_cpu_v4_warpAffine
    cpp-version/rotate_image_v4_warpAffine.cpp
    cpp-version/util.cpp
)




# Include directories
target_include_directories(image_rotation_gray PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(image_rotation_gray PRIVATE
    ${OpenCV_LIBS}
    cudart
    OpenMP::OpenMP_CXX
)

# CUDA compilation properties
set_target_properties(image_rotation_gray PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


# Include directories
target_include_directories(image_rotation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(image_rotation PRIVATE
    ${OpenCV_LIBS}
    cudart
    OpenMP::OpenMP_CXX
)

# CUDA compilation properties
set_target_properties(image_rotation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(rotate_image_cpu_v1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rotate_image_cpu_v2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rotate_image_cpu_v3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rotate_image_cpu_v4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rotate_image_cpu_v4_warpAffine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(rotate_image_cpu_v2 PRIVATE ${OpenCV_LIBS})
target_link_libraries(rotate_image_cpu_v3 PRIVATE ${OpenCV_LIBS})
target_link_libraries(rotate_image_cpu_v4 PRIVATE ${OpenCV_LIBS})
target_link_libraries(rotate_image_cpu_v4_warpAffine PRIVATE ${OpenCV_LIBS})

# Optional: install target
install(TARGETS image_rotation image_rotation_gray rotate_image_cpu_v1 rotate_image_cpu_v2 rotate_image_cpu_v3 rotate_image_cpu_v4 rotate_image_cpu_v4_warpAffine DESTINATION bin)

