#pragma once

#include <climits>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace grpo {
    enum class AdvantageMode{
        standardized,
        centered
    };

    enum class ReductionMode{
        sequence_mean,
        token_mean,
        length_weighted
    };

    struct LossConfig{
        float clip_eps=0.2f;
        float beta=0.0f;
        ReductionMode reduction=ReductionMode::sequence_mean;
        float length_alpha=0.0f;
    };

    struct LossStats{
        float loss=0.0f;
        float pg_loss=0.0f;
        float kl=0.0f;
        int valid_tokens=0;
    };

    struct LossResult{
        LossStats stats;
        std::vector<float> dlogp_new;
    };

    namespace detail {
        struct Shape{
            std::size_t sequences;
            std::size_t tokens;
        };

        inline Shape checked_shape(int B, int G, int T){
            if(B<=0 || G<=0 || T<=0) throw std::runtime_error("B, G and T must be positive");
            auto b=static_cast<std::size_t>(B);
            auto g=static_cast<std::size_t>(G);
            auto t=static_cast<std::size_t>(T);
            if(b>std::numeric_limits<std::size_t>::max()/g)
                throw std::runtime_error("sequence count is too large");
            auto sequences=b*g;
            if(sequences>static_cast<std::size_t>(INT_MAX))
                throw std::runtime_error("sequence count exceeds INT_MAX");
            if(sequences>std::numeric_limits<std::size_t>::max()/t)
                throw std::runtime_error("token count is too large");
            auto tokens=sequences*t;
            if(tokens>static_cast<std::size_t>(INT_MAX))
                throw std::runtime_error("token count exceeds INT_MAX");
            return {sequences,tokens};
        }
    }

    inline std::size_t idx2(int b, int g, int G){
        return static_cast<std::size_t>(b)*G+g;
    }
    
    inline std::size_t idx3(int b, int g, int t, int G, int T){
        return (static_cast<std::size_t>(b)*G+g)*T+t;
    }

    std::vector<float> group_advantages_cpu(
        const std::vector<float>& rewards,
        int B,
        int G,
        AdvantageMode mode=AdvantageMode::standardized,
        float eps=1e-8f
    );

    LossResult grpo_loss_cpu(
        const std::vector<float>& logp_new,
        const std::vector<float>& logp_old,
        const std::vector<float>& logp_ref,
        const std::vector<float>& advantages,
        const std::vector<int>& mask,
        int B,
        int G,
        int T,
        LossConfig config={}
    );
}
