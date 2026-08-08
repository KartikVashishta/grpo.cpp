#include "grpo/vigor.hpp"

#include <algorithm>
#include <cmath>
#include <climits>
#include <numeric>
#include <stdexcept>

namespace grpo {
    VigorAllocator::VigorAllocator(int prompts, VigorConfig config)
        : config_(config), rollouts_per_prompt_(config.initial_rollouts),
          active_(prompts>0 ? prompts : 0), rewards_(prompts>0 ? prompts : 0),
          observed_round_(prompts>0 ? prompts : 0,-1){
        if(prompts<=0) throw std::runtime_error("prompts must be positive");
        if(config.initial_rollouts<=0 || config.rounds<=0)
            throw std::runtime_error("VIGOR rollout counts must be positive");
        if(!std::isfinite(config.selection_ratio) ||
           config.selection_ratio<=0.0f || config.selection_ratio>1.0f)
            throw std::runtime_error("VIGOR selection ratio must be in (0, 1]");
        if(config.expansion<=1)
            throw std::runtime_error("VIGOR expansion must be greater than one");
        std::iota(active_.begin(),active_.end(),0);
    }

    void VigorAllocator::observe(int prompt, const std::vector<float>& rewards){
        if(prompt<0 || prompt>=static_cast<int>(rewards_.size()))
            throw std::runtime_error("VIGOR prompt index is out of range");
        if(std::find(active_.begin(),active_.end(),prompt)==active_.end())
            throw std::runtime_error("cannot observe an inactive VIGOR prompt");
        if(observed_round_[prompt]==round_)
            throw std::runtime_error("VIGOR prompt was observed twice in one round");
        if(static_cast<int>(rewards.size())!=rollouts_per_prompt_)
            throw std::runtime_error("wrong number of VIGOR rewards for this round");
        for(float reward:rewards){
            if(!std::isfinite(reward))
                throw std::runtime_error("VIGOR rewards must be finite");
        }
        rewards_[prompt].insert(rewards_[prompt].end(),rewards.begin(),rewards.end());
        observed_round_[prompt]=round_;
    }

    float VigorAllocator::reward_variance(int prompt) const{
        if(prompt<0 || prompt>=static_cast<int>(rewards_.size()))
            throw std::runtime_error("VIGOR prompt index is out of range");
        const auto& values=rewards_[prompt];
        if(values.empty()) return 0.0f;
        double mean=0.0;
        for(float value:values) mean+=value;
        mean/=static_cast<double>(values.size());
        double variance=0.0;
        for(float value:values){
            double delta=static_cast<double>(value)-mean;
            variance+=delta*delta;
        }
        return static_cast<float>(variance/static_cast<double>(values.size()));
    }

    int VigorAllocator::rollout_count(int prompt) const{
        if(prompt<0 || prompt>=static_cast<int>(rewards_.size()))
            throw std::runtime_error("VIGOR prompt index is out of range");
        return static_cast<int>(rewards_[prompt].size());
    }

    bool VigorAllocator::refine(){
        for(int prompt:active_){
            if(observed_round_[prompt]!=round_)
                throw std::runtime_error("every active VIGOR prompt must be observed");
        }
        if(round_+1>=config_.rounds) return false;

        std::stable_sort(active_.begin(),active_.end(),[&](int a, int b){
            float va=reward_variance(a);
            float vb=reward_variance(b);
            if(va!=vb) return va>vb;
            return a<b;
        });
        size_t keep=static_cast<size_t>(std::ceil(
            static_cast<double>(active_.size())*config_.selection_ratio
        ));
        active_.resize(std::max<size_t>(1,keep));
        if(rollouts_per_prompt_>INT_MAX/config_.expansion)
            throw std::runtime_error("VIGOR rollout count is too large");
        rollouts_per_prompt_*=config_.expansion;
        round_++;
        return true;
    }
}
