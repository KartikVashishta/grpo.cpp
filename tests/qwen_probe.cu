#include "model/qwen3.h"
#include "cuda_utils.h"
#include "third_party/json.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;

template <typename T>
void save_pool(const std::filesystem::path& path, const T* device, size_t count) {
    std::vector<T> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.write(reinterpret_cast<const char*>(host.data()), count * sizeof(T));
}

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error("usage: qwen_probe MODEL FIXTURE OUTPUT SCALE MICROBATCH");
        const auto fixture = json::parse(std::ifstream(argv[2]));
        const std::filesystem::path output(argv[3]);
        const float scale = std::stof(argv[4]);
        const int microbatch = std::stoi(argv[5]);
        const int B = fixture.at("batch"), S = fixture.at("sequence");
        auto tokens = fixture.at("tokens").get<std::vector<int>>();
        auto targets = fixture.at("targets").get<std::vector<int>>();
        auto upstream = fixture.at("upstream").get<std::vector<float>>();
        if (B != 2 || S != 193 || (microbatch != 1 && microbatch != 2) ||
            !std::isfinite(scale) || scale <= 0 || tokens.size() != B*S ||
            targets.size() != tokens.size() || upstream.size() != tokens.size())
            throw std::runtime_error("Invalid fixture dimensions or loss scale");
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i] < 0 || tokens[i] >= 151936 || targets[i] < 0 || targets[i] >= 151936 ||
                !std::isfinite(upstream[i] * scale))
                throw std::runtime_error("Invalid token or coefficient");
            upstream[i] *= scale;
        }
        if (!std::filesystem::create_directory(output))
            throw std::runtime_error("Output directory already exists");

        auto* model = qwen3_load(argv[1], B, S);
        qwen3_sleep(model);
        auto* state = qwen3_train_state_alloc(model->config, B, S);
        auto* grads = qwen3_gradients_alloc(model->config, B*S);
        float *d_logp, *d_upstream;
        CUDA_CHECK(cudaMalloc(&d_logp, B*S*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_upstream, B*S*sizeof(float)));
        qwen3_gradients_zero(grads);
        std::vector<float> logp(B*S);
        for (int b = 0; b < B; b += microbatch) {
            qwen3_forward(model, state, tokens.data()+b*S, targets.data()+b*S,
                          d_logp, microbatch, S);
            CUDA_CHECK(cudaMemcpy(logp.data()+b*S, d_logp, microbatch*S*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(d_upstream, upstream.data()+b*S, microbatch*S*sizeof(float), cudaMemcpyHostToDevice));
            qwen3_backward(model, state, grads, d_upstream);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        save_pool(output / "half.bin", grads->half_pool, model->weights.fp16_pool_elems);
        save_pool(output / "float.bin", grads->float_pool, model->weights.fp32_pool_elems);
        std::ofstream(output / "logp.json") << json(logp).dump() << '\n';
        qwen3_gradients_free(grads);
        qwen3_train_state_free(state);
        qwen3_free(model);
        CUDA_CHECK(cudaFree(d_logp));
        CUDA_CHECK(cudaFree(d_upstream));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
