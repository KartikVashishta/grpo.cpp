#pragma once

#include <vector>

namespace grpo {
    struct VigorConfig{
        int initial_rollouts=2;
        int rounds=4;
        float selection_ratio=0.5f;
        int expansion=2;
    };

    class VigorAllocator{
    public:
        explicit VigorAllocator(int prompts, VigorConfig config={});

        const std::vector<int>& active_prompts() const { return active_; }
        int rollouts_per_prompt() const { return rollouts_per_prompt_; }
        int round() const { return round_; }

        void observe(int prompt, const std::vector<float>& rewards);
        bool refine();

        float reward_variance(int prompt) const;
        int rollout_count(int prompt) const;

    private:
        VigorConfig config_;
        int round_=0;
        int rollouts_per_prompt_=0;
        std::vector<int> active_;
        std::vector<std::vector<float>> rewards_;
        std::vector<int> observed_round_;
    };
}
