cmake_minimum_required(VERSION 3.24)
project(grpo_cpp LANGUAGES CXX)

add_library(grpo_cpu
    src/cpu/grpo_loss.cpp
    src/cpu/vigor.cpp
)
target_include_directories(grpo_cpu PUBLIC include)
target_compile_features(grpo_cpu PUBLIC cxx_std_20)

add_executable(grpo_loss_cpu apps/00_grpo_loss_cpu.cpp)
target_link_libraries(grpo_loss_cpu PRIVATE grpo_cpu)

add_executable(tiny_grpo_train apps/01_tiny_grpo_train.cpp)
target_link_libraries(tiny_grpo_train PRIVATE grpo_cpu)

add_executable(length_tradeoff apps/02_length_tradeoff.cpp)
target_link_libraries(length_tradeoff PRIVATE grpo_cpu)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_grpo_loss tests/test_grpo_loss.cpp)
    target_link_libraries(test_grpo_loss PRIVATE grpo_cpu)
    add_test(NAME grpo_loss_cpu COMMAND test_grpo_loss)
    add_test(NAME tiny_grpo_train COMMAND tiny_grpo_train)
    add_test(NAME length_tradeoff COMMAND length_tradeoff)
endif()

option(GRPO_USE_CUDA "Build CUDA labs and kernels" OFF)
option(GRPO_BUILD_GPT2 "Build the optional llm.c GPT-2 example" OFF)
option(GRPO_BUILD_DISTRIBUTED "Build the optional two-GPU GPT-2 example" OFF)

if(GRPO_BUILD_GPT2 AND NOT GRPO_USE_CUDA)
    message(FATAL_ERROR "GRPO_BUILD_GPT2 requires GRPO_USE_CUDA=ON")
endif()
if(GRPO_BUILD_DISTRIBUTED AND NOT GRPO_BUILD_GPT2)
    message(FATAL_ERROR "GRPO_BUILD_DISTRIBUTED requires GRPO_BUILD_GPT2=ON")
endif()

if(GRPO_USE_CUDA)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    add_library(grpo_cuda src/cuda/grpo_loss_cuda.cu)
    target_include_directories(grpo_cuda PUBLIC include)
    target_link_libraries(grpo_cuda PUBLIC grpo_cpu)

    add_executable(bench_grpo_loss_cuda bench/bench_grpo_loss_cuda.cu)
    target_link_libraries(bench_grpo_loss_cuda PRIVATE grpo_cuda)
    target_compile_definitions(bench_grpo_loss_cuda PRIVATE
        GRPO_BUILD_CONFIG="$<CONFIG>"
    )

    find_package(Threads REQUIRED)
    add_executable(two_gpu_grpo_train apps/03_two_gpu_train.cu)
    target_link_libraries(two_gpu_grpo_train PRIVATE grpo_cuda Threads::Threads)

    add_executable(lab_00_vector_add labs/00_vector_add.cu)
    add_executable(lab_01_reduce_sum labs/01_reduce_sum.cu)

    set_target_properties(
        lab_00_vector_add
        lab_01_reduce_sum
        grpo_cuda
        bench_grpo_loss_cuda
        two_gpu_grpo_train
        PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    target_include_directories(lab_00_vector_add PRIVATE include)
    target_include_directories(lab_01_reduce_sum PRIVATE include)

    if(GRPO_BUILD_GPT2)
        include(FetchContent)
        FetchContent_Declare(
            llmc
            GIT_REPOSITORY https://github.com/karpathy/llm.c.git
            GIT_TAG f1e2ace651495b74ae22d45d1723443fd00ecd3a
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(llmc)
        find_package(CUDAToolkit REQUIRED)

        add_executable(gpt2_grpo apps/04_gpt2_grpo.cu)
        target_include_directories(gpt2_grpo PRIVATE ${llmc_SOURCE_DIR})
        target_link_libraries(gpt2_grpo PRIVATE grpo_cuda CUDA::cublas)
        set_target_properties(gpt2_grpo PROPERTIES
            # llm.c defines its own lerp; C++20 adds std::lerp and newer nvcc
            # considers the unqualified call ambiguous.
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )

        if(GRPO_BUILD_DISTRIBUTED)
            find_path(NCCL_INCLUDE_DIR nccl.h REQUIRED)
            find_library(NCCL_LIBRARY nccl REQUIRED)
            add_executable(gpt2_grpo_2gpu apps/05_two_gpu_gpt2_grpo.cu)
            target_include_directories(gpt2_grpo_2gpu PRIVATE
                ${llmc_SOURCE_DIR}
                ${NCCL_INCLUDE_DIR}
            )
            target_link_libraries(gpt2_grpo_2gpu PRIVATE
                grpo_cuda CUDA::cublas ${NCCL_LIBRARY}
            )
            set_target_properties(gpt2_grpo_2gpu PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED ON
            )
        endif()
    endif()

    if(BUILD_TESTING)
        add_test(NAME grpo_loss_cuda
            COMMAND bench_grpo_loss_cuda --check-only
        )
        add_test(NAME two_gpu_grpo_train
            COMMAND two_gpu_grpo_train
        )
        set_tests_properties(two_gpu_grpo_train PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
