#pragma once

#include "grpo/grpo_loss.hpp"

namespace grpo {
    enum class CudaLossKernel{
        atomic,
        block_reduce
    };

    struct CudaTiming{
        float atomic_ms=0.0f;
        float block_ms=0.0f;
    };

    LossResult grpo_loss_cuda(
        const std::vector<float>& logp_new,
        const std::vector<float>& logp_old,
        const std::vector<float>& logp_ref,
        const std::vector<float>& advantages,
        const std::vector<int>& mask,
        int B,
        int G,
        int T,
        LossConfig config={},
        CudaLossKernel kernel=CudaLossKernel::block_reduce
    );

    CudaTiming benchmark_grpo_loss_cuda(
        const std::vector<float>& logp_new,
        const std::vector<float>& logp_old,
        const std::vector<float>& logp_ref,
        const std::vector<float>& advantages,
        const std::vector<int>& mask,
        int B,
        int G,
        int T,
        LossConfig config={},
        int warmup=10,
        int iterations=100
    );

    LogitsLossResult grpo_logits_cuda(
        const std::vector<float>& logits_new,
        const std::vector<float>& logp_old,
        const std::vector<float>& logp_ref,
        const std::vector<int>& selected_tokens,
        const std::vector<float>& advantages,
        const std::vector<int>& mask,
        int B,
        int G,
        int T,
        int V,
        LossConfig config={}
    );
}
