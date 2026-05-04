#pragma once

#include <vector>

namespace grpo {
    struct LossStats{
        float loss=0.0f;
        float pg_loss=0.0f; // policy gradient loss (rho)
        float kl=0.0f; // exp(change) - change - 1
        int valid_tokens=0; // number of valid tokens excluding padding/masked tokens
    };

    inline int idx2(int b, int g, int G){
        return b*G+g;
    }
    
    inline int idx3(int b, int g, int t, int G, int T){
        return (b*G+g)*T+t;
    }

    std::vector<float> group_advantages_cpu(
        const std::vector<float>& rewards,
        int B,
        int G,
        float eps=1e-8f
    );

    LossStats grpo_loss_cpu(
        const std::vector<float>& logp_new,
        const std::vector<float>& logp_old,
        const std::vector<float>& logp_ref,
        const std::vector<float>& advantages,
        const std::vector<int>& mask,
        int B,
        int G,
        int T,
        float clip_eps,
        float beta
    );
}