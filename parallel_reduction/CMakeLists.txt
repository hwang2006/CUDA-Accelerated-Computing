add_executable(reduction
    main.cpp
    reduction.cu
)

target_include_directories(reduction PRIVATE
    
    
)

target_link_libraries(reduction PRIVATE
    cudart
    OpenMP::OpenMP_CXX
)

set_target_properties(reduction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

install(TARGETS reduction DESTINATION bin)
