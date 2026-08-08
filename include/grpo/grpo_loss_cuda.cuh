#pragma once

#include "grpo/grpo_loss.hpp"

namespace grpo {
    enum class CudaLossKernel{
        atomic,
        block_reduce
    };

    enum class CudaLogitsKernel{
        separate,
        fused
    };

    struct CudaTiming{
        float atomic_ms=0.0f;
        float block_ms=0.0f;
    };

    struct CudaLogitsTiming{
        float separate_ms=0.0f;
        float fused_ms=0.0f;
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
        LossConfig config={},
        CudaLogitsKernel kernel=CudaLogitsKernel::fused
    );

    // logits_and_dlogits points to B*G*T rows of P device floats. The first V
    // entries form the vocabulary and the optional padded tail is ignored. Logits
    // are replaced in place by dL/dlogits; the smaller rollout metadata lives
    // on the host and is copied for this call.
    LossStats grpo_logits_cuda_device_inplace(
        float* logits_and_dlogits,
        const std::vector<float>& logp_old,
        const std::vector<float>& logp_ref,
        const std::vector<int>& selected_tokens,
        const std::vector<float>& advantages,
        const std::vector<int>& mask,
        int B,
        int G,
        int T,
        int V,
        int P,
        LossConfig config={},
        CudaLogitsKernel kernel=CudaLogitsKernel::fused
    );

    CudaLogitsTiming benchmark_grpo_logits_cuda(
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
        LossConfig config={},
        int warmup=10,
        int iterations=100
    );
}
