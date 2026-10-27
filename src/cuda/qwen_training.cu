#include "grpo/qwen_training.cuh"
#include "third_party/json.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace grpo {
namespace {
template <typename T>
__global__ void sum_squares(const T* values, size_t count, float inverse_scale, float* total) {
    __shared__ float sums[256];
    float sum = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const float value = static_cast<float>(values[i]) * inverse_scale;
        sum += value * value;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = 128; offset; offset /= 2) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(total, sums[0]);
}

struct Tensor {
    std::string name;
    std::vector<size_t> shape;
    const float* data;
    size_t count;
};

std::vector<Tensor> tensors(const Qwen3Model& model, AdamWOptimizer& optimizer) {
    const auto& c = model.config;
    const size_t h = c.hidden_size, q = c.q_dim, kv = c.kv_dim, mid = c.intermediate_size;
    const size_t head = c.head_dim;
    const float* matrix = optimizer.fp16_master();
    const float* norm = model.weights.fp32_pool;
    std::vector<Tensor> result;
    auto add = [&](const std::string& name, std::vector<size_t> shape, const float*& pool) {
        size_t count = 1;
        for (size_t dim : shape) count *= dim;
        result.push_back({name + ".weight", std::move(shape), pool, count});
        pool += count;
    };
    for (int i = 0; i < c.num_hidden_layers; ++i) {
        const auto p = "model.layers." + std::to_string(i) + ".";
        add(p + "self_attn.q_proj", {q, h}, matrix);
        add(p + "self_attn.k_proj", {kv, h}, matrix);
        add(p + "self_attn.v_proj", {kv, h}, matrix);
        add(p + "self_attn.o_proj", {h, q}, matrix);
        add(p + "mlp.gate_proj", {mid, h}, matrix);
        add(p + "mlp.up_proj", {mid, h}, matrix);
        add(p + "mlp.down_proj", {h, mid}, matrix);
        add(p + "input_layernorm", {h}, norm);
        add(p + "self_attn.q_norm", {head}, norm);
        add(p + "self_attn.k_norm", {head}, norm);
        add(p + "post_attention_layernorm", {h}, norm);
    }
    add("model.embed_tokens", {static_cast<size_t>(c.vocab_size), h}, matrix);
    add("model.norm", {h}, norm);
    if (matrix - optimizer.fp16_master() != optimizer.num_fp16_params() ||
        norm - model.weights.fp32_pool != optimizer.num_fp32_params())
        throw std::runtime_error("Checkpoint tensor layout does not match the weight pools");
    return result;
}
}

float qwen_step(AdamWOptimizer& optimizer, Qwen3Gradients& grads,
                float loss_scale, float max_norm, cudaStream_t stream) {
    if (!std::isfinite(loss_scale) || loss_scale < 1 || !std::isfinite(max_norm) || max_norm <= 0 ||
        !std::isfinite(optimizer.lr) || optimizer.lr <= 0 ||
        !(optimizer.beta1 >= 0 && optimizer.beta1 < 1) ||
        !(optimizer.beta2 >= 0 && optimizer.beta2 < 1) ||
        !std::isfinite(optimizer.eps) || optimizer.eps <= 0 ||
        !std::isfinite(optimizer.weight_decay) || optimizer.weight_decay < 0)
        throw std::runtime_error("Invalid loss scale, clipping norm or Adam settings");
    float* device_sum;
    CUDA_CHECK(cudaMalloc(&device_sum, sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(device_sum, 0, sizeof(float), stream));
    // The remaining float gradient pool is attention scratch, not parameters.
    sum_squares<<<1024, 256, 0, stream>>>(grads.half_pool, optimizer.num_fp16_params(), 1.0f / loss_scale, device_sum);
    sum_squares<<<256, 256, 0, stream>>>(grads.float_pool, optimizer.num_fp32_params(), 1.0f / loss_scale, device_sum);
    float sum;
    CUDA_CHECK(cudaMemcpyAsync(&sum, device_sum, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(device_sum));
    const float norm = std::sqrt(sum);
    if (!std::isfinite(norm)) throw std::runtime_error("Nonfinite gradient; no optimizer update made");
    const float multiplier = std::min(1.0f, max_norm / (norm + 1e-6f)) / loss_scale;
    optimizer.step(&grads, stream, multiplier);
    CUDA_CHECK(cudaGetLastError());
    return norm;
}

void save_qwen_checkpoint(const std::filesystem::path& output,
                          const std::filesystem::path& source,
                          const Qwen3Model& model, AdamWOptimizer& optimizer,
                          cudaStream_t stream) {
    using json = nlohmann::json;
    const auto pending = std::filesystem::path(output.string() + ".tmp");
    if (std::filesystem::exists(output) || !std::filesystem::create_directory(pending))
        throw std::runtime_error("Checkpoint already exists: " + output.string());
    try {
        const auto views = tensors(model, optimizer);
        json header = {{"__metadata__", {{"format", "pt"}}}};
        uint64_t offset = 0;
        for (const auto& tensor : views) {
            header[tensor.name] = {{"dtype", "F32"}, {"shape", tensor.shape},
                {"data_offsets", {offset, offset + tensor.count * sizeof(float)}}};
            offset += tensor.count * sizeof(float);
        }
        std::string text = header.dump();
        text.append((8 - text.size() % 8) % 8, ' ');
        std::ofstream weights(pending / "model.safetensors", std::ios::binary);
        weights.exceptions(std::ios::failbit | std::ios::badbit);
        const uint64_t length = text.size();
        for (int byte = 0; byte < 8; ++byte) weights.put(static_cast<char>(length >> (8 * byte)));
        weights.write(text.data(), text.size());
        std::vector<float> chunk(1 << 20);
        for (const auto& tensor : views) {
            for (size_t start = 0; start < tensor.count; start += chunk.size()) {
                const size_t count = std::min(chunk.size(), tensor.count - start);
                CUDA_CHECK(cudaMemcpyAsync(chunk.data(), tensor.data + start, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (!std::all_of(chunk.begin(), chunk.begin() + count, [](float x) { return std::isfinite(x); }))
                    throw std::runtime_error("Nonfinite checkpoint weight: " + tensor.name);
                weights.write(reinterpret_cast<const char*>(chunk.data()), count * sizeof(float));
            }
        }
        weights.close();
        auto config = json::parse(std::ifstream(source / "config.json"));
        config["torch_dtype"] = "float32";
        config["dtype"] = "float32";
        std::ofstream config_file(pending / "config.json");
        config_file.exceptions(std::ios::failbit | std::ios::badbit);
        config_file << config.dump(2) << '\n';
        config_file.close();
        for (const auto& name : {"tokenizer.json", "tokenizer_config.json", "generation_config.json",
                                 "special_tokens_map.json", "chat_template.jinja"}) {
            if (std::filesystem::is_regular_file(source / name))
                std::filesystem::copy_file(source / name, pending / name);
        }
        std::filesystem::rename(pending, output);
    } catch (...) {
        std::filesystem::remove_all(pending);
        throw;
    }
}
}
