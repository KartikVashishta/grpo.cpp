#include "grpo/grpo_loss.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {
    constexpr int prompts=3;
    constexpr int actions=4;
    constexpr int states=6;
    constexpr int group_size=16;
    constexpr int max_length=3;
    constexpr int outer_updates=100;
    constexpr int passes=4;
    constexpr float learning_rate=0.5f;

    struct Rollout{
        std::array<int,max_length> state{};
        std::array<int,max_length> action{};
        int length=0;
        float reward=0;
    };

    std::array<float,actions> probabilities(const std::vector<float>& logits, int state){
        std::array<float,actions> p{};
        float high=logits[state*actions];
        for(int a=1;a<actions;a++) high=std::max(high,logits[state*actions+a]);
        float sum=0;
        for(int a=0;a<actions;a++){
            p[a]=std::exp(logits[state*actions+a]-high);
            sum+=p[a];
        }
        for(float& value:p) value/=sum;
        return p;
    }

    float action_logp(const std::vector<float>& logits, int state, int action){
        auto p=probabilities(logits,state);
        return std::log(p[action]);
    }

    Rollout sample_rollout(
        const std::vector<float>& logits,
        int prompt,
        std::mt19937& rng
    ){
        Rollout rollout;
        int state=prompt;
        for(int t=0;t<max_length;t++){
            auto p=probabilities(logits,state);
            std::discrete_distribution<int> pick(p.begin(),p.end());
            int action=pick(rng);
            rollout.state[t]=state;
            rollout.action[t]=action;
            rollout.length++;
            if(action==3) break;
            state=prompts+action;
        }
        rollout.reward=(
            rollout.length==2 &&
            rollout.action[0]==prompt &&
            rollout.action[1]==3
        ) ? 1.0f : 0.0f;
        return rollout;
    }

    float exact_success(const std::vector<float>& logits, int prompt){
        auto first=probabilities(logits,prompt);
        auto second=probabilities(logits,prompts+prompt);
        return first[prompt]*second[3];
    }

    bool greedy_is_correct(const std::vector<float>& logits, int prompt){
        auto first=probabilities(logits,prompt);
        int first_action=std::max_element(first.begin(),first.end())-first.begin();
        if(first_action!=prompt) return false;
        auto second=probabilities(logits,prompts+first_action);
        int second_action=std::max_element(second.begin(),second.end())-second.begin();
        return second_action==3;
    }

    struct RunResult{
        float initial=0;
        float final=0;
        bool greedy=false;
        bool finite=true;
    };

    RunResult train(unsigned seed){
        std::mt19937 rng(seed);
        std::vector<float> logits(states*actions,0.0f);
        std::vector<float> reference=logits;
        RunResult run;
        for(int p=0;p<prompts;p++) run.initial+=exact_success(logits,p)/prompts;

        grpo::LossConfig config;
        config.clip_eps=0.2f;
        config.beta=0.01f;
        config.reduction=grpo::ReductionMode::sequence_mean;

        int B=prompts,G=group_size,T=max_length;
        int n_sequences=B*G;
        int n_tokens=n_sequences*T;

        for(int update=0;update<outer_updates;update++){
            auto old_logits=logits;
            std::vector<Rollout> rollouts(n_sequences);
            std::vector<float> rewards(n_sequences);
            std::vector<float> logp_old(n_tokens,0);
            std::vector<float> logp_ref(n_tokens,0);
            std::vector<int> mask(n_tokens,0);

            for(int p=0;p<prompts;p++){
                for(int g=0;g<group_size;g++){
                    int seq=grpo::idx2(p,g,G);
                    rollouts[seq]=sample_rollout(old_logits,p,rng);
                    rewards[seq]=rollouts[seq].reward;
                    for(int t=0;t<rollouts[seq].length;t++){
                        int i=seq*T+t;
                        int state=rollouts[seq].state[t];
                        int action=rollouts[seq].action[t];
                        logp_old[i]=action_logp(old_logits,state,action);
                        logp_ref[i]=action_logp(reference,state,action);
                        mask[i]=1;
                    }
                }
            }

            auto advantages=grpo::group_advantages_cpu(
                rewards,B,G,grpo::AdvantageMode::standardized
            );

            for(int pass=0;pass<passes;pass++){
                std::vector<float> logp_new(n_tokens,0);
                for(int seq=0;seq<n_sequences;seq++){
                    for(int t=0;t<rollouts[seq].length;t++){
                        int i=seq*T+t;
                        logp_new[i]=action_logp(
                            logits,rollouts[seq].state[t],rollouts[seq].action[t]
                        );
                    }
                }

                auto result=grpo::grpo_loss_cpu(
                    logp_new,logp_old,logp_ref,advantages,mask,B,G,T,config
                );
                std::vector<float> gradient(logits.size(),0.0f);
                for(int seq=0;seq<n_sequences;seq++){
                    for(int t=0;t<rollouts[seq].length;t++){
                        int i=seq*T+t;
                        int state=rollouts[seq].state[t];
                        int chosen=rollouts[seq].action[t];
                        auto p=probabilities(logits,state);
                        for(int action=0;action<actions;action++){
                            float dlogp=(action==chosen ? 1.0f : 0.0f)-p[action];
                            gradient[state*actions+action]+=result.dlogp_new[i]*dlogp;
                        }
                    }
                }

                for(size_t i=0;i<logits.size();i++){
                    logits[i]-=learning_rate*gradient[i];
                    if(!std::isfinite(logits[i]) || !std::isfinite(gradient[i])) run.finite=false;
                }
                if(!std::isfinite(result.stats.loss)) run.finite=false;
            }
        }

        run.greedy=true;
        for(int p=0;p<prompts;p++){
            run.final+=exact_success(logits,p)/prompts;
            run.greedy=run.greedy && greedy_is_correct(logits,p);
        }
        return run;
    }
}

int main(){
    std::array<float,5> finals{};
    bool passed=true;
    std::cout << std::fixed << std::setprecision(6);
    for(unsigned seed=0;seed<finals.size();seed++){
        auto run=train(seed);
        finals[seed]=run.final;
        std::cout << "seed " << seed
                  << ": initial=" << run.initial
                  << " final=" << run.final
                  << " greedy=" << (run.greedy ? "yes" : "no")
                  << " finite=" << (run.finite ? "yes" : "no") << "\n";
        passed=passed && std::fabs(run.initial-0.0625f)<1e-6f;
        passed=passed && run.final>0.90f && run.greedy && run.finite;
    }
    std::sort(finals.begin(),finals.end());
    std::cout << "median final=" << finals[2] << "\n";
    passed=passed && finals[2]>0.95f;
    if(!passed){
        std::cerr << "training did not converge\n";
        return 1;
    }
}
