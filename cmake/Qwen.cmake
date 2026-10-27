include(FetchContent)
find_package(CUDAToolkit REQUIRED)
find_program(PATCH_EXECUTABLE patch REQUIRED)

# Use the model and kernels; RL.cu's trainer has a different loss and no saver.
FetchContent_Declare(rlcu
    URL https://codeload.github.com/KJLdefeated/RL.cu/tar.gz/77ecd8385d9836bc8377e042f4e2b7dd0b6df9c6
    URL_HASH SHA256=87a3f71cabc6f56db0eb9bbba5a1aca6630804e3988a9410f199dfe58d44fc2f
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR grpo-unused
    PATCH_COMMAND ${PATCH_EXECUTABLE} -p1 -i ${CMAKE_CURRENT_LIST_DIR}/patches/rlcu-weights.patch
)
FetchContent_Declare(qwen_json
    URL https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp
    URL_HASH SHA256=9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6
    DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(rlcu qwen_json)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/qwen_include/third_party)
configure_file(${qwen_json_SOURCE_DIR}/json.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/qwen_include/third_party/json.hpp COPYONLY)

set(qwen_sources src/model/qwen3.cu src/model/kv_cache.cu)
foreach(kernel rmsnorm rope attention swiglu embedding linear softmax sampler adamw fused_norm_linear)
    list(APPEND qwen_sources src/kernels/${kernel}.cu)
endforeach()
list(APPEND qwen_sources src/kernels/config.cpp src/kernels/weights.cpp)
list(TRANSFORM qwen_sources PREPEND ${rlcu_SOURCE_DIR}/)
add_library(qwen_cuda STATIC ${qwen_sources})
target_include_directories(qwen_cuda PUBLIC
    ${rlcu_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/qwen_include)
target_link_libraries(qwen_cuda PUBLIC CUDA::cublas CUDA::cudart)
target_compile_features(qwen_cuda PUBLIC cxx_std_17)
target_compile_options(qwen_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(qwen_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

add_executable(qwen_probe tests/qwen_probe.cu)
target_link_libraries(qwen_probe PRIVATE qwen_cuda)
set_target_properties(qwen_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
