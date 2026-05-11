option(GRPO_USE_CUDA "Build CUDA labs and kernels" OFF)

if(GRPO_USE_CUDA)
    enable_language(CUDA)

    add_executable(lab_00_vector_add
        labs/00_vector_add.cu
    )

    add_executable(lab_01_reduce_sum
        labs/01_reduce_sum.cu
    )

    set_target_properties(
        lab_00_vector_add
        lab_01_reduce_sum
        PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
endif()