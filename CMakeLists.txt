cmake_minimum_required(VERSION 3.24)
project(grpo_cpp LANGUAGES CXX)

add_library(grpo_cpu
    src/cpu/grpo_loss.cpp
    src/cpu/vigor.cpp
)
target_include_directories(grpo_cpu PUBLIC include)
target_compile_features(grpo_cpu PUBLIC cxx_std_20)

add_executable(tiny_grpo_train examples/tiny_grpo_train.cpp)
target_link_libraries(tiny_grpo_train PRIVATE grpo_cpu)

option(GRPO_BUILD_EXPERIMENTS "Build the research experiments" OFF)
if(GRPO_BUILD_EXPERIMENTS)
    add_executable(length_weighting experiments/length_weighting.cpp)
    target_link_libraries(length_weighting PRIVATE grpo_cpu)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test_grpo_loss tests/test_grpo_loss.cpp)
    target_link_libraries(test_grpo_loss PRIVATE grpo_cpu)
    add_test(NAME grpo_loss COMMAND test_grpo_loss)
    add_test(NAME tiny_grpo_train COMMAND tiny_grpo_train)
    if(GRPO_BUILD_EXPERIMENTS)
        add_test(NAME length_weighting COMMAND length_weighting)
    endif()
endif()

option(GRPO_USE_CUDA "Build the CUDA kernels and benchmark" OFF)
option(GRPO_BUILD_GPT2 "Build the optional llm.c GPT-2 example" OFF)
option(GRPO_BUILD_QWEN "Build the Qwen3-0.6B trainer and checks" OFF)
option(GRPO_BUILD_DISTRIBUTED "Build the optional two-GPU GPT-2 example" OFF)

if(GRPO_BUILD_GPT2 AND NOT GRPO_USE_CUDA)
    message(FATAL_ERROR "GRPO_BUILD_GPT2 requires GRPO_USE_CUDA=ON")
endif()
if(GRPO_BUILD_QWEN AND NOT GRPO_USE_CUDA)
    message(FATAL_ERROR "GRPO_BUILD_QWEN requires GRPO_USE_CUDA=ON")
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

    set_target_properties(
        grpo_cuda
        bench_grpo_loss_cuda
        PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

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

        add_executable(gpt2_grpo examples/gpt2_grpo.cu)
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
            add_executable(gpt2_grpo_2gpu examples/gpt2_grpo_2gpu.cu)
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
    endif()
    if(GRPO_BUILD_QWEN)
        include(cmake/Qwen.cmake)
    endif()
endif()
