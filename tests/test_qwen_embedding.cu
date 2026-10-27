#include "kernels/embedding.cuh"
#include "cuda_utils.h"

#include <array>
#include <iostream>

int main() {
    constexpr int T = 5, H = 4, V = 4;
    const std::array<int, T> ids{1, 1, 1, 2, 2};
    const std::array<float, T> values{2048, 1, -2048, 0.5, -0.25};
    std::array<half, T*H> input;
    std::array<half, V*H> weights;
    weights.fill(__float2half(0.25f));
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) input[t*H+h] = __float2half(values[t]);
    half *d_input, *d_weights;
    int* d_ids;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(input)));
    CUDA_CHECK(cudaMalloc(&d_weights, sizeof(weights)));
    CUDA_CHECK(cudaMalloc(&d_ids, sizeof(ids)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), sizeof(input), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), sizeof(weights), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), sizeof(ids), cudaMemcpyHostToDevice));
    for (int step = 1; step <= 2; ++step) {
        launch_embedding_backward(d_weights, d_input, d_ids, T, H);
        launch_embedding_backward(d_weights, d_input, d_ids, 0, H);
        CUDA_CHECK(cudaMemcpy(weights.data(), d_weights, sizeof(weights), cudaMemcpyDeviceToHost));
        for (int v = 0; v < V; ++v) {
            const float expected = 0.25f + step * (v == 1 ? 1.0f : v == 2 ? 0.25f : 0.0f);
            for (int h = 0; h < H; ++h) {
                if (__half2float(weights[v*H+h]) != expected) {
                    std::cerr << "Wrong embedding gradient at row " << v << '\n';
                    return 1;
                }
            }
        }
    }
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_ids));
    std::cout << "Repeated-token gradients preserve the existing head contribution\n";
}
