#include "kernels/sampler.cuh"
#include "cuda_utils.h"
#include <cstdio>
#include <vector>

int main() {
    constexpr int V = 151936;
    std::vector<half> logits(V, __float2half(-1000));
    logits[0] = __float2half(0);
    half* device;
    int64_t* output;
    CUDA_CHECK(cudaMalloc(&device, V*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&output, sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(device, logits.data(), V*sizeof(half), cudaMemcpyHostToDevice));
    // At seed 93, token 92530's uniform rounds to 1 without the upper clamp.
    for (auto seed : {93ULL, 0xcf9a04affa6badc0ULL}) {
        launch_sampler(device, nullptr, 1, V, 0, 1, 1, seed, output);
        int64_t token;
        CUDA_CHECK(cudaMemcpy(&token, output, sizeof(token), cudaMemcpyDeviceToHost));
        if (token != 0) {
            std::fprintf(stderr, "Expected token 0, got %ld for seed %llu\n", long(token), seed);
            return 1;
        }
    }
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(device));
    std::puts("sampler endpoints passed");
}
