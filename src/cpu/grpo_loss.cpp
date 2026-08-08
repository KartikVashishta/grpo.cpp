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
        AdvantageMode mode,
        float eps
    ){
        auto shape=detail::checked_shape(B,G,1);
        if(eps<0.0f || !std::isfinite(eps)) throw std::runtime_error("eps must be finite and non-negative");
        expect_size("rewards", rewards.size(),shape.sequences);
        for(float reward:rewards){
            if(!std::isfinite(reward)) throw std::runtime_error("rewards must be finite");
        }
        std::vector<float> advantages(shape.sequences,0.0f);

        for(int b=0; b<B; b++){
            float first=rewards[idx2(b,0,G)];
            bool all_equal=true;
            for(int g=1; g<G; g++){
                all_equal=all_equal && rewards[idx2(b,g,G)]==first;
            }
            if(all_equal) continue;

            double mean=0.0;
            double m2=0.0;
            for(int g=0; g<G; g++){
                double r=rewards[idx2(b,g,G)];
                double d=r-mean;
                mean+=d/static_cast<double>(g+1);
                m2+=d*(r-mean);
            }
            double var=m2/static_cast<double>(G);
            double denom=mode==AdvantageMode::standardized ? std::sqrt(var+eps) : 1.0;
            for(int g=0; g<G; g++){
                double r=rewards[idx2(b,g,G)];
                advantages[idx2(b,g,G)]=static_cast<float>((r-mean)/denom);
            }
        }
        for(float advantage:advantages){
            if(!std::isfinite(advantage)) throw std::runtime_error("advantages became non-finite");
        }
        return advantages;
    }

    LossResult grpo_loss_cpu(
        const std::vector<float>& logp_new,
        const std::vector<float>& logp_old,
        const std::vector<float>& logp_ref,
        const std::vector<float>& advantages,
        const std::vector<int>& mask,
        int B,
        int G,
        int T,
        LossConfig config
    ){
        auto shape=detail::checked_shape(B,G,T);
        if(config.clip_eps<0.0f || config.clip_eps>=1.0f || !std::isfinite(config.clip_eps))
            throw std::runtime_error("clip_eps must be finite and in [0, 1)");
        if(config.beta<0.0f || !std::isfinite(config.beta))
            throw std::runtime_error("beta must be finite and non-negative");
        if(!std::isfinite(config.length_alpha) || config.length_alpha<0.0f || config.length_alpha>1.0f)
            throw std::runtime_error("length_alpha must be finite and in [0, 1]");

        const size_t n_tokens=shape.tokens;
        const size_t n_seq=shape.sequences;

        expect_size("logp_new", logp_new.size(), n_tokens);
        expect_size("logp_old", logp_old.size(), n_tokens);
        expect_size("logp_ref", logp_ref.size(), n_tokens);
        expect_size("advantages", advantages.size(), n_seq);
        expect_size("mask", mask.size(), n_tokens);

        for(size_t i=0; i<n_tokens; i++){
            if(!std::isfinite(logp_new[i]) || !std::isfinite(logp_old[i]) || !std::isfinite(logp_ref[i]))
                throw std::runtime_error("log probabilities must be finite");
            if(mask[i]!=0 && mask[i]!=1) throw std::runtime_error("mask values must be 0 or 1");
        }
        for(float advantage:advantages){
            if(!std::isfinite(advantage)) throw std::runtime_error("advantages must be finite");
        }

        std::vector<int> lengths(n_seq,0);
        int valid_tokens=0;

        for(size_t i=0; i<n_tokens; i++){
            if(mask[i]==0) continue;
            lengths[i/static_cast<size_t>(T)]++;
            valid_tokens++;
        }

        for(int length:lengths){
            if(length==0) throw std::runtime_error("every sequence needs at least one valid token");
        }

        LossResult result;
        result.stats.valid_tokens=valid_tokens;
        result.dlogp_new.assign(n_tokens,0.0f);

        double total_loss=0.0;
        double total_pg_loss=0.0;
        double total_kl=0.0;

        for(int b=0; b<B; b++){
            for(int g=0; g<G; g++){
                int seq = idx2(b,g,G);
                float A = advantages[idx2(b,g,G)];
                float weight=0.0f;

                if(config.reduction==ReductionMode::sequence_mean){
                    weight=1.0f/(static_cast<float>(n_seq)*lengths[seq]);
                }else if(config.reduction==ReductionMode::token_mean){
                    weight=1.0f/static_cast<float>(valid_tokens);
                }else{
                    weight=std::pow(static_cast<float>(lengths[seq]),config.length_alpha-1.0f)
                        /static_cast<float>(n_seq);
                }

                for (int t=0; t<T; t++){
                    int i = idx3(b,g,t,G,T);
                    if(mask[i]==0) continue;

                    float pg_loss=0.0f;
                    float pg_grad=0.0f;
                    if(A!=0.0f){
                        float rho=std::exp(logp_new[i]-logp_old[i]);
                        float clipped_rho=std::clamp(rho,1.0f-config.clip_eps,1.0f+config.clip_eps);
                        pg_loss=-std::min(rho*A,clipped_rho*A);

                        // At the exact boundary we keep the unclipped subgradient.
                        bool clipped_high=A>0.0f && rho>1.0f+config.clip_eps;
                        bool clipped_low=A<0.0f && rho<1.0f-config.clip_eps;
                        pg_grad=clipped_high || clipped_low ? 0.0f : -A*rho;
                    }

                    float d = logp_ref[i]-logp_new[i];
                    float expm1_d = std::expm1(d);
                    float kl_approx = expm1_d-d;

                    float loss = pg_loss + config.beta*kl_approx;
                    float kl_grad = -config.beta*expm1_d;

                    total_loss+=weight*loss;
                    total_pg_loss+=weight*pg_loss;
                    total_kl+=weight*kl_approx;
                    result.dlogp_new[i]=weight*(pg_grad+kl_grad);
                }
            }
        }

        result.stats.loss=static_cast<float>(total_loss);
        result.stats.pg_loss=static_cast<float>(total_pg_loss);
        result.stats.kl=static_cast<float>(total_kl);
        if(!std::isfinite(result.stats.loss) || !std::isfinite(result.stats.pg_loss) || !std::isfinite(result.stats.kl))
            throw std::runtime_error("loss became non-finite");
        for(float grad:result.dlogp_new){
            if(!std::isfinite(grad)) throw std::runtime_error("gradient became non-finite");
        }
        return result;
    }
}
