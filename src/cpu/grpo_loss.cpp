#include "grpo/grpo_loss.hpp"

#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace grpo {

    // helper procedure to help us inspect the size differences if any
    static void expect_size(const char* name, size_t got, size_t want){
        if(got!=want)
            throw std::runtime_error(std::string(name) + " has size " + std::to_string(got) + ", expected " +std::to_string(want));
    }

    std::vector<float> group_advantages_cpu(
        const std::vector<float>& rewards,
        int B, 
        int G,
        float eps
    ){
        if(B<=0||G<=0) throw std::runtime_error("B and G must be positive");
        expect_size("rewards", rewards.size(),static_cast<size_t>(B*G));
        std::vector<float> advantages(B*G,0.0f);

        // for advantage we need the avg and the std deviations
        // avg is the sum of reward per generation / G
        // std is root ((sum of each (reward-avg)^2) + eta)
        // once that's calculated, for each reward, we do ri - avg / std
        for(int b=0; b<B; b++){
            float mean = 0.0f;
            float var = 0.0f;
            for(int g=0; g<G; g++){
                mean+=rewards[idx2(b,g,G)];
            }
            mean/=G;
            for(int g=0; g<G; g++){
                float r=rewards[idx2(b,g,G)];
                float d=r-mean;
                var+=d*d;
            }
            var/=static_cast<float>(G);
            float denom = std::sqrt(var+eps);
            for(int g=0; g<G; g++){
                float r=rewards[idx2(b,g,G)];
                advantages[idx2(b,g,G)]=(r-mean)/denom;
            }
        }
        return advantages;
    }

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
    ){
        if(B<=0 || G<=0 || T<=0) throw std::runtime_error("B, G and T must be positive");

        const size_t n_tokens=static_cast<size_t>(B)*G*T;
        const size_t n_seq=static_cast<size_t>(B)*G;

        expect_size("logp_new", logp_new.size(), n_tokens);
        expect_size("logp_old", logp_old.size(), n_tokens);
        expect_size("logp_ref", logp_ref.size(), n_tokens);
        expect_size("advantages", advantages.size(), n_seq);
        expect_size("mask", mask.size(), n_tokens);

        double total_loss=0.0;
        double total_pg_loss=0.0;
        double total_kl=0.0;
        int valid_tokens=0;

        for(int b=0; b<B; b++){
            for(int g=0; g<G; g++){
                float A = advantages[idx2(b,g,G)];
                for (int t=0; t<T; t++){
                    int i = idx3(b,g,t,G,T);
                    if(mask[i]==0) continue;
                    // the loss function is surrogate - beta*KL divergence
                    // the surrogate is min(clip(rho,1-e,1+e)*A, rho*A)
                    float rho = std::exp(logp_new[i]-logp_old[i]);
                    float clipped_rho = std::clamp(rho, 1.0f-clip_eps, 1.0f+clip_eps);
                    float surrogate = std::min(clipped_rho*A, rho*A);

                    // the kl is log(del)-del-1
                    // del = ref[i]-new[i]
                    float d = logp_ref[i]-logp_new[i];
                    float kl_approx = std::exp(d)-d-1.0f;

                    float pg_loss = -surrogate;
                    float loss = pg_loss + beta * kl_approx;
                    total_loss+=loss;
                    total_pg_loss+=pg_loss;
                    total_kl+=kl_approx;
                    valid_tokens++;
                }
            }
        }
        LossStats stats;
        stats.valid_tokens=valid_tokens;

        if(valid_tokens>0){
            stats.loss=static_cast<float>(total_loss/valid_tokens);
            stats.pg_loss=static_cast<float>(total_pg_loss/valid_tokens);
            stats.kl=static_cast<float>(total_kl/valid_tokens);
        }

        return stats;
    }
}
