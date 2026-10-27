#include "grpo/grpo_loss_cuda.cuh"
#include "grpo/qwen_training.cuh"
#include "kernels/sampler.cuh"
#include "cuda_utils.h"
#include "third_party/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <unistd.h>

using json = nlohmann::json;
using Tokens = std::vector<std::vector<int>>;
using Clock = std::chrono::steady_clock;

static double seconds(Clock::time_point start) {
    CUDA_CHECK(cudaDeviceSynchronize());
    return std::chrono::duration<double>(Clock::now() - start).count();
}

class Trainer {
    static constexpr int batch_limit = 32, sequence_limit = 256, microbatch = 2;
    static constexpr float loss_scale = 256;
    std::string source;
    Qwen3Model* model = nullptr;
    Qwen3TrainState* state = nullptr;
    Qwen3Gradients* grads = nullptr;
    std::unique_ptr<AdamWOptimizer> optimizer;
    float *d_logp = nullptr, *d_upstream = nullptr;
    int64_t* d_sample = nullptr;
    Tokens prompts, completions;
    bool pending_update = false;
    size_t peak_used = 0;

    void check_tokens(const Tokens& rows) const {
        if (rows.empty() || rows.size() > batch_limit)
            throw std::runtime_error("Expected 1 to 32 token sequences");
        for (const auto& row : rows) {
            if (row.empty() || row.size() > sequence_limit)
                throw std::runtime_error("Expected 1 to 256 tokens per sequence");
            for (int id : row)
                if (id < 0 || id >= model->config.vocab_size)
                    throw std::runtime_error("Token outside vocabulary");
        }
    }

    void memory() {
        size_t available, total;
        CUDA_CHECK(cudaMemGetInfo(&available, &total));
        peak_used = std::max(peak_used, total - available);
    }

    std::vector<float> group_advantages(const Tokens& input, const Tokens& output,
                                       const std::vector<float>& rewards, int G) const {
        check_tokens(input);
        check_tokens(output);
        if (G < 2 || rewards.size() != input.size() || output.size() != input.size() || rewards.size() % G)
            throw std::runtime_error("Expected grouped rewards for each completion");
        for (size_t i = 0; i < rewards.size(); ++i)
            if (!std::isfinite(rewards[i]) || input[i] != input[(i/G)*G])
                throw std::runtime_error("Each reward group must share one prompt");
        return grpo::group_advantages_cpu(rewards, rewards.size()/G, G, grpo::AdvantageMode::centered);
    }

    json score(const Tokens& input, const Tokens& output, const std::vector<float>& advantages = {}) {
        check_tokens(input);
        check_tokens(output);
        if (input.size() != output.size()) throw std::runtime_error("Mismatched sequence counts");
        int S = 0;
        for (size_t i = 0; i < input.size(); ++i)
            S = std::max(S, int(input[i].size() + output[i].size() - 1));
        if (S > sequence_limit) throw std::runtime_error("Prompt and completion exceed 256 positions");
        const bool backward = !advantages.empty();
        if (backward) qwen3_gradients_zero(grads);
        json logps = json::array();
        double forward_seconds = 0, backward_seconds = 0, loss_seconds = 0;
        float loss = 0;
        for (size_t start = 0; start < input.size(); start += microbatch) {
            const int B = std::min(size_t(microbatch), input.size() - start);
            std::vector<int> tokens(B*S, 0), targets(B*S, 0), mask(B*S, 0);
            for (int b = 0; b < B; ++b) {
                auto full = input[start+b];
                full.insert(full.end(), output[start+b].begin(), output[start+b].end());
                std::copy(full.begin(), full.end()-1, tokens.begin()+b*S);
                std::copy(full.begin()+1, full.end(), targets.begin()+b*S);
                std::fill(mask.begin()+b*S+input[start+b].size()-1,
                          mask.begin()+b*S+full.size()-1, 1);
            }
            auto clock = Clock::now();
            qwen3_forward(model, state, tokens.data(), targets.data(), d_logp, B, S);
            std::vector<float> values(B*S);
            CUDA_CHECK(cudaMemcpy(values.data(), d_logp, values.size()*sizeof(float), cudaMemcpyDeviceToHost));
            for (int b = 0; b < B; ++b) {
                std::vector<float> row;
                for (int t = 0; t < S; ++t) if (mask[b*S+t]) {
                    if (!std::isfinite(values[b*S+t])) throw std::runtime_error("Nonfinite token probability");
                    row.push_back(values[b*S+t]);
                }
                logps.push_back(row);
            }
            forward_seconds += seconds(clock);
            if (!backward) continue;
            clock = Clock::now();
            std::vector<float> adv(advantages.begin()+start, advantages.begin()+start+B);
            // One pass per rollout: old and current weights are still identical.
            // beta=0 needs no reference model; use current logps for that input.
            auto result = grpo::grpo_loss_cuda(values, values, values, adv, mask, B, 1, S);
            const float batch_fraction = float(B) / input.size();
            loss += result.stats.loss * batch_fraction;
            for (float& value : result.dlogp_new) value *= batch_fraction * loss_scale;
            CUDA_CHECK(cudaMemcpy(d_upstream, result.dlogp_new.data(), B*S*sizeof(float), cudaMemcpyHostToDevice));
            loss_seconds += seconds(clock);
            clock = Clock::now();
            qwen3_backward(model, state, grads, d_upstream);
            backward_seconds += seconds(clock);
            memory();
        }
        return {{"logps", logps}, {"loss", loss}, {"stage_seconds", {
            {"scoring_forward", forward_seconds}, {"loss", loss_seconds},
            {"backward_and_batch_overhead", backward_seconds}}}};
    }

    json generate(const json& request) {
        auto input = request.at("prompts").get<Tokens>();
        check_tokens(input);
        const int cap = request.at("max_new_tokens");
        const float temperature = request.at("temperature");
        const auto eos = request.at("eos_ids").get<std::vector<int>>();
        auto seed = request.at("seed").get<unsigned long long>();
        const bool trace = request.value("trace", false);
        if (cap < 1 || cap > 64 || !std::isfinite(temperature) || temperature < 0 || temperature > 2 || eos.empty())
            throw std::runtime_error("Invalid sampling settings");
        for (int id : eos) if (id < 0 || id >= model->config.vocab_size)
            throw std::runtime_error("EOS outside vocabulary");
        int blocks = 0;
        for (const auto& row : input) {
            if (row.size() + cap > sequence_limit) throw std::runtime_error("Generation exceeds 256 positions");
            blocks += (row.size() + cap + 15) / 16;
        }
        auto clock = Clock::now();
        qwen3_wakeup(model, blocks);
        memory();
        std::vector<Sequence> sequences;
        sequences.reserve(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            sequences.emplace_back(i, std::vector<int64_t>(input[i].begin(), input[i].end()), SamplingParams{});
            sequences.back().batch_slot = i;
        }
        std::vector<Sequence*> active;
        for (auto& sequence : sequences) active.push_back(&sequence);
        std::vector<std::vector<double>> logps(input.size());
        std::vector<json> top_tokens(input.size(), json::array());
        std::vector<int> active_sizes;
        for (int t = 0; t < cap && !active.empty(); ++t) {
            active_sizes.push_back(active.size());
            half* logits = t == 0 ? qwen3_prefill(model, active) : qwen3_decode(model, active);
            launch_sampler(logits, nullptr, active.size(), model->config.vocab_size, 0, 1,
                           temperature, seed, d_sample);
            seed += active.size();
            std::vector<int64_t> ids(active.size());
            CUDA_CHECK(cudaMemcpy(ids.data(), d_sample, ids.size()*sizeof(int64_t), cudaMemcpyDeviceToHost));
            if (trace) {
                // This copy is for the inference check, and is disabled during training.
                const int V = model->config.vocab_size;
                std::vector<half> host(active.size()*V);
                CUDA_CHECK(cudaMemcpy(host.data(), logits, host.size()*sizeof(half), cudaMemcpyDeviceToHost));
                for (size_t b = 0; b < active.size(); ++b) {
                    double maximum = -INFINITY, sum = 0;
                    int first = -1, second = -1;
                    for (int v = 0; v < V; ++v) {
                        const double value = __half2float(host[b*V+v]);
                        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite inference logit");
                        if (value > maximum) {
                            second = first;
                            first = v;
                            maximum = value;
                        } else if (second < 0 || value > __half2float(host[b*V+second])) second = v;
                    }
                    for (int v = 0; v < V; ++v) sum += std::exp(double(__half2float(host[b*V+v])) - maximum);
                    const double selected = __half2float(host[b*V+ids[b]]);
                    if (temperature == 0 && selected != maximum)
                        throw std::runtime_error("Greedy sampler did not select a maximum logit");
                    const double value = selected - maximum - std::log(sum);
                    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite inference probability");
                    logps[active[b]->batch_slot].push_back(value);
                    top_tokens[active[b]->batch_slot].push_back({{"tokens", {first, second}},
                        {"logits", {maximum, double(__half2float(host[b*V+second]))}},
                        {"selected_logit", selected}});
                }
            }
            for (size_t b = 0; b < active.size(); ++b) active[b]->append_token(ids[b]);
            active.erase(std::remove_if(active.begin(), active.end(), [&](Sequence* sequence) {
                return std::find(eos.begin(), eos.end(), sequence->last_token_id) != eos.end();
            }), active.end());
        }
        Tokens result;
        for (auto& sequence : sequences) {
            auto ids = sequence.completion_token_ids();
            result.emplace_back(ids.begin(), ids.end());
        }
        qwen3_sleep(model);
        prompts = std::move(input);
        completions = result;
        pending_update = temperature == 1.0f;
        json response = {{"completions", result}, {"next_seed", seed},
                         {"seconds", seconds(clock)}, {"active_sizes", active_sizes}};
        if (trace) {
            response["logps"] = logps;
            response["top_tokens"] = top_tokens;
        }
        return response;
    }

public:
    Trainer(const std::string& path, float learning_rate) : source(path) {
        if (!std::isfinite(learning_rate) || learning_rate <= 0)
            throw std::runtime_error("Learning rate must be positive");
        cudaDeviceProp device{};
        CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
        if (device.major < 8 || device.sharedMemPerBlockOptin < 96704)
            throw std::runtime_error("Qwen needs SM80 or newer and 96,704 bytes of shared memory per block");
        std::cerr << "GPU: " << device.name << ", SM" << device.major << device.minor
                  << ", " << device.totalGlobalMem << " bytes, " << device.sharedMemPerBlockOptin
                  << " bytes shared memory per block\n";
        model = new Qwen3Model{};
        model->config = load_config(source);
        std::vector<float> master;
        model->weights = load_weights(source, model->config, &master);
        qwen3_init(model, batch_limit, sequence_limit, batch_limit*sequence_limit, 1);
        qwen3_sleep(model);
        optimizer = std::make_unique<AdamWOptimizer>(model, learning_rate, .9f, .999f, 1e-8f, 0, master.data());
        state = qwen3_train_state_alloc(model->config, microbatch, sequence_limit);
        grads = qwen3_gradients_alloc(model->config, microbatch*sequence_limit);
        CUDA_CHECK(cudaMalloc(&d_logp, microbatch*sequence_limit*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_upstream, microbatch*sequence_limit*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sample, batch_limit*sizeof(int64_t)));
        memory();
    }

    ~Trainer() {
        cudaFree(d_sample);
        cudaFree(d_upstream);
        cudaFree(d_logp);
        qwen3_gradients_free(grads);
        qwen3_train_state_free(state);
        optimizer.reset();
        qwen3_free(model);
    }

    json handle(const json& request) {
        const std::string command = request.at("command");
        json result;
        if (command == "generate") result = generate(request);
        else if (command == "score")
            result = score(request.at("prompts").get<Tokens>(), request.at("completions").get<Tokens>());
        else if (command == "benchmark") {
            const auto input = request.at("prompts").get<Tokens>();
            const auto output = request.at("completions").get<Tokens>();
            const auto rewards = request.at("rewards").get<std::vector<float>>();
            const auto advantages = group_advantages(input, output, rewards, request.at("group_size"));
            const int warmup = request.at("warmup"), iterations = request.at("iterations");
            if (warmup < 1 || warmup > 10 || iterations < 1 || iterations > 30)
                throw std::runtime_error("Use 1–10 warmups and 1–30 benchmark iterations");
            json samples = json::array();
            for (int i = 0; i < warmup + iterations; ++i) {
                auto clock = Clock::now();
                result = score(input, output, advantages);
                const double elapsed = seconds(clock);
                if (i >= warmup) samples.push_back({{"seconds", elapsed},
                    {"stage_seconds", result.at("stage_seconds")}, {"loss", result.at("loss")}});
            }
            result["samples"] = samples;
            result["gradient_norm"] = grpo::qwen_gradient_norm(*optimizer, *grads, loss_scale);
            result["optimizer_step"] = optimizer->get_step();
        } else if (command == "update") {
            const auto rewards = request.at("rewards").get<std::vector<float>>();
            const int G = request.at("group_size");
            if (!pending_update)
                throw std::runtime_error("Update needs grouped rewards from a temperature-1 generation");
            const auto advantages = group_advantages(prompts, completions, rewards, G);
            result = score(prompts, completions, advantages);
            auto clock = Clock::now();
            result["gradient_norm"] = grpo::qwen_step(*optimizer, *grads, loss_scale);
            result["stage_seconds"]["optimizer"] = seconds(clock);
            result["step"] = optimizer->get_step();
            result["advantages"] = advantages;
            pending_update = false;
        } else if (command == "save") {
            auto clock = Clock::now();
            grpo::save_qwen_checkpoint(request.at("path").get<std::string>(), source, *model, *optimizer);
            result = {{"seconds", seconds(clock)}, {"step", optimizer->get_step()}};
        } else throw std::runtime_error("Unknown command: " + command);
        memory();
        result["peak_device_used_bytes"] = peak_used;
        return result;
    }
};

int main(int argc, char** argv) {
    FILE* responses = nullptr;
    try {
        if (argc != 3) throw std::runtime_error("usage: qwen_train MODEL LEARNING_RATE");
        responses = fdopen(dup(STDOUT_FILENO), "w");
        if (!responses || dup2(STDERR_FILENO, STDOUT_FILENO) < 0)
            throw std::runtime_error("Cannot open response stream");
        Trainer trainer(argv[1], std::stof(argv[2]));
        std::fputs("{\"ready\":true}\n", responses);
        std::fflush(responses);
        std::string line;
        while (std::getline(std::cin, line)) {
            auto result = trainer.handle(json::parse(line));
            std::fputs((result.dump() + "\n").c_str(), responses);
            if (std::fflush(responses) != 0) throw std::runtime_error("Cannot write response");
        }
        std::fclose(responses);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (responses) {
            std::fputs((json({{"error", error.what()}}).dump()+"\n").c_str(), responses);
            std::fclose(responses);
        }
        return 1;
    }
}
