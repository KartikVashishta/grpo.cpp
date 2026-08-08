cmake_minimum_required(VERSION 3.24)
project(grpo_cpp LANGUAGES CXX)

add_library(grpo_cpu src/cpu/grpo_loss.cpp)
target_include_directories(grpo_cpu PUBLIC include)
target_compile_features(grpo_cpu PUBLIC cxx_std_20)

add_executable(grpo_loss_cpu apps/00_grpo_loss_cpu.cpp)
target_link_libraries(grpo_loss_cpu PRIVATE grpo_cpu)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_grpo_loss tests/test_grpo_loss.cpp)
    target_link_libraries(test_grpo_loss PRIVATE grpo_cpu)
    add_test(NAME grpo_loss_cpu COMMAND test_grpo_loss)
endif()

option(GRPO_USE_CUDA "Build CUDA labs and kernels" OFF)

if(GRPO_USE_CUDA)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    add_executable(lab_00_vector_add labs/00_vector_add.cu)
    add_executable(lab_01_reduce_sum labs/01_reduce_sum.cu)

    set_target_properties(
        lab_00_vector_add
        lab_01_reduce_sum
        PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    target_include_directories(lab_00_vector_add PRIVATE include)
    target_include_directories(lab_01_reduce_sum PRIVATE include)
endif()
