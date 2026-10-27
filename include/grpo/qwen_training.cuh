#pragma once

#include "training/optimizer.h"
#include <filesystem>

namespace grpo {
float qwen_gradient_norm(AdamWOptimizer& optimizer, Qwen3Gradients& grads,
                         float loss_scale, cudaStream_t stream = 0);

float qwen_step(AdamWOptimizer& optimizer, Qwen3Gradients& grads,
                float loss_scale, float max_norm = 1.0f, cudaStream_t stream = 0);

// Model-only HF checkpoint. Optimizer moments and the step counter are not saved.
void save_qwen_checkpoint(const std::filesystem::path& output,
                          const std::filesystem::path& source,
                          const Qwen3Model& model, AdamWOptimizer& optimizer,
                          cudaStream_t stream = 0);
}
