#include "grpo/qwen_training.cuh"
#include "third_party/json.hpp"

#include <fstream>
#include <iostream>
#include <limits>

using json = nlohmann::json;

template <typename T>
std::vector<float> read(const T* pointer, size_t count) {
    std::vector<T> data(count);
    CUDA_CHECK(cudaMemcpy(data.data(), pointer, count*sizeof(T), cudaMemcpyDeviceToHost));
    return std::vector<float>(data.begin(), data.end());
}

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    Qwen3Model model{};
    auto& c = model.config;
    c.hidden_size = 4; c.intermediate_size = 8; c.num_hidden_layers = 2;
    c.num_attention_heads = 2; c.num_key_value_heads = 1; c.head_dim = 2; c.vocab_size = 7;
    c.compute_derived();
    const size_t n16 = c.num_hidden_layers * c.hidden_size * (2*c.q_dim + 2*c.kv_dim + 3*c.intermediate_size)
                     + c.vocab_size * c.hidden_size;
    const size_t n32 = c.num_hidden_layers * (2*c.hidden_size + 2*c.head_dim) + c.hidden_size;
    model.weights.fp16_pool_elems = n16; model.weights.fp32_pool_elems = n32;
    CUDA_CHECK(cudaMalloc(&model.weights.fp16_pool, n16*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&model.weights.fp32_pool, n32*sizeof(float)));
    std::vector<float> master(n16), norms(n32, 0.125f);
    std::vector<half> weights(n16);
    for (size_t i = 0; i < n16; ++i) {
        master[i] = i % 4 == 0 ? 0.0f : 0.1f + i * 1e-5f;
        weights[i] = __float2half(master[i]);
    }
    CUDA_CHECK(cudaMemcpy(model.weights.fp16_pool, weights.data(), n16*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(model.weights.fp32_pool, norms.data(), n32*sizeof(float), cudaMemcpyHostToDevice));
    AdamWOptimizer optimizer(&model, 1e-6f, 0.9f, 0.999f, 1e-8f, 0.0f, master.data());
    auto* grads = qwen3_gradients_alloc(c, 7);
    qwen3_gradients_zero(grads);
    CUDA_CHECK(cudaMemset(grads->D_buf, 0xff, 7*c.num_attention_heads*sizeof(float)));
    json result = {{"initial_matrix", master}, {"initial_norm", norms}, {"steps", json::array()}};
    auto state = [&]() {
        return json{{"matrix", read(optimizer.fp16_master(), n16)}, {"norm", read(model.weights.fp32_pool, n32)},
            {"effective_matrix", read(model.weights.fp16_pool, n16)},
            {"matrix_m", read(optimizer.fp16_m(), n16)}, {"matrix_v", read(optimizer.fp16_v(), n16)},
            {"norm_m", read(optimizer.fp32_m(), n32)}, {"norm_v", read(optimizer.fp32_v(), n32)},
            {"step", optimizer.get_step()}};
    };
    for (int step = 0; step < 3; ++step) {
        std::vector<half> half_grad(n16);
        std::vector<float> float_grad(n32);
        for (size_t i = 0; i < n16; ++i)
            half_grad[i] = __float2half(step == 2 ? 0.0f : i % 4 == 0 ? 0x1p-20f : (static_cast<int>(i % 7)-3) * (step == 0 ? 64.f : -32.f));
        for (size_t i = 0; i < n32; ++i)
            float_grad[i] = step == 2 ? 0.0f : (static_cast<int>(i % 5)-2) * (step == 0 ? 128.f : -64.f);
        CUDA_CHECK(cudaMemcpy(grads->half_pool, half_grad.data(), n16*sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(grads->float_pool, float_grad.data(), n32*sizeof(float), cudaMemcpyHostToDevice));
        const float norm = grpo::qwen_step(optimizer, *grads, 256.f, step == 1 ? 100.f : 1.f);
        auto row = state();
        row["gradient_norm"] = norm;
        row["max_norm"] = step == 1 ? 100.f : 1.f;
        row["matrix_grad"] = read(grads->half_pool, n16);
        row["norm_grad"] = float_grad;
        result["steps"].push_back(row);
    }
    const auto before = state();
    int rejected = 0;
    const half nan = __float2half(std::numeric_limits<float>::quiet_NaN());
    CUDA_CHECK(cudaMemcpy(grads->half_pool, &nan, sizeof(half), cudaMemcpyHostToDevice));
    try { grpo::qwen_step(optimizer, *grads, 256.f); } catch (const std::runtime_error&) { ++rejected; }
    qwen3_gradients_zero(grads);
    const float infinity = std::numeric_limits<float>::infinity();
    CUDA_CHECK(cudaMemcpy(grads->float_pool, &infinity, sizeof(float), cudaMemcpyHostToDevice));
    try { grpo::qwen_step(optimizer, *grads, 256.f); } catch (const std::runtime_error&) { ++rejected; }
    result["nonfinite_rejected_without_changes"] = rejected == 2 && state() == before;
    if (!result["nonfinite_rejected_without_changes"].get<bool>()) return 1;
    std::ofstream out(argv[1]);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << result.dump(2) << '\n';
    out.close();
    qwen3_gradients_free(grads);
    CUDA_CHECK(cudaFree(model.weights.fp16_pool));
    CUDA_CHECK(cudaFree(model.weights.fp32_pool));
    std::cout << "Optimizer fixture saved\n";
}
