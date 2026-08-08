#include "grpo/grpo_loss.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// L^(alpha-1) from https://arxiv.org/abs/2607.23364

namespace {
    constexpr int groups=100000;
    constexpr int group_size=8;
    constexpr int max_length=8;
    constexpr double continue_p=0.7;

    struct Moment{
        int n=0;
        double mean=0;
        double m2=0;

        void add(double x){
            n++;
            double delta=x-mean;
            mean+=delta/n;
            m2+=delta*(x-mean);
        }

        double standard_error() const{
            return std::sqrt((m2/(n-1))/n);
        }
    };

    double exact_expected_update(double alpha){
        double reward_mean=0;
        double weighted_score_mean=0;
        double reward_score_mean=0;

        for(int length=1;length<=max_length;length++){
            double probability=std::pow(continue_p,length-1)*(1.0-continue_p);
            double score=(length-1)*(1.0-continue_p)-continue_p;
            double weight=std::pow(static_cast<double>(length),alpha-1.0);
            double reward=length>=5 ? 1.0 : 0.0;
            reward_mean+=probability*reward;
            weighted_score_mean+=probability*weight*score;
            reward_score_mean+=probability*reward*weight*score;
        }

        // The other length-eight outcome is eight consecutive continues.
        double probability=std::pow(continue_p,max_length);
        double score=max_length*(1.0-continue_p);
        double weight=std::pow(static_cast<double>(max_length),alpha-1.0);
        reward_mean+=probability;
        weighted_score_mean+=probability*weight*score;
        reward_score_mean+=probability*weight*score;
        return reward_score_mean-reward_mean*weighted_score_mean;
    }

    std::array<Moment,3> monte_carlo(){
        std::mt19937 rng(12345);
        std::bernoulli_distribution continues(continue_p);
        std::array<Moment,3> moments;
        std::array<float,3> alphas={0.0f,0.5f,1.0f};

        int n_tokens=group_size*max_length;
        std::vector<float> now(n_tokens),old(n_tokens),ref(n_tokens);
        std::vector<float> score(n_tokens);
        std::vector<float> rewards(group_size);
        std::vector<int> mask(n_tokens);

        for(int sample=0;sample<groups;sample++){
            std::fill(mask.begin(),mask.end(),0);
            for(int g=0;g<group_size;g++){
                int length=0;
                for(int t=0;t<max_length;t++){
                    bool action_continue=continues(rng);
                    int i=g*max_length+t;
                    mask[i]=1;
                    length++;
                    float logp=std::log(action_continue ? continue_p : 1.0-continue_p);
                    now[i]=old[i]=ref[i]=logp;
                    score[i]=action_continue ? 1.0f-continue_p : -continue_p;
                    if(!action_continue) break;
                }
                rewards[g]=length>=5 ? 1.0f : 0.0f;
            }

            auto advantages=grpo::group_advantages_cpu(
                rewards,1,group_size,grpo::AdvantageMode::centered
            );
            for(size_t a=0;a<alphas.size();a++){
                grpo::LossConfig config;
                config.beta=0;
                config.reduction=grpo::ReductionMode::length_weighted;
                config.length_alpha=alphas[a];
                auto result=grpo::grpo_loss_cpu(
                    now,old,ref,advantages,mask,1,group_size,max_length,config
                );
                double loss_gradient=0;
                for(int i=0;i<n_tokens;i++){
                    if(mask[i]) loss_gradient+=result.dlogp_new[i]*score[i];
                }
                double reward_gradient=-loss_gradient*group_size/(group_size-1.0);
                moments[a].add(reward_gradient);
            }
        }
        return moments;
    }
}

int main(){
    std::array<double,3> alphas={0.0,0.5,1.0};
    auto moments=monte_carlo();
    bool passed=true;
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "alpha  expected update  monte carlo  std. error\n";
    for(size_t i=0;i<alphas.size();i++){
        double exact=exact_expected_update(alphas[i]);
        std::cout << std::setw(4) << alphas[i] << "   "
                  << exact << "   " << moments[i].mean << "   "
                  << moments[i].standard_error() << "\n";
        passed=passed && std::fabs(moments[i].mean-exact)<0.003;
    }

    std::cout << "\nnominal loss-weight share (length 2 vs 8)\n";
    for(double alpha:alphas){
        double short_mass=std::pow(2.0,alpha);
        double long_mass=std::pow(8.0,alpha);
        double share=long_mass/(short_mass+long_mass);
        std::cout << "alpha " << alpha << ": " << 100.0*share << "%\n";
    }

    if(!passed){
        std::cerr << "length check failed\n";
        return 1;
    }
}
