#include "grpo/grpo_loss.hpp"
#include "grpo/grpo_loss_cuda.cuh"
#include "grpo/cuda_utils.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

namespace {
    constexpr int prompts=3;
    constexpr int actions=4;
    constexpr int states=6;
    constexpr int num_devices=2;
    constexpr int local_group=8;
    constexpr int global_group=num_devices*local_group;
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

    struct Shard{
        std::vector<Rollout> rollouts;
        std::vector<float> logp_old;
        std::vector<float> logp_ref;
        std::vector<float> advantages;
        std::vector<int> mask;
        int loss_calls=0;
        long long valid_tokens=0;
        int nonzero_calls=0;
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
        return std::log(probabilities(logits,state)[action]);
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

    void sample_group(
        const std::vector<float>& old_logits,
        const std::vector<float>& reference,
        std::array<std::mt19937,num_devices>& rng,
        std::array<Shard,num_devices>& shards
    ){
        int local_sequences=prompts*local_group;
        int local_tokens=local_sequences*max_length;
        std::vector<float> global_rewards(prompts*global_group,0.0f);

        for(int device_id=0;device_id<num_devices;device_id++){
            auto& shard=shards[device_id];
            shard.rollouts.assign(local_sequences,{});
            shard.logp_old.assign(local_tokens,0.0f);
            shard.logp_ref.assign(local_tokens,0.0f);
            shard.advantages.assign(local_sequences,0.0f);
            shard.mask.assign(local_tokens,0);

            for(int p=0;p<prompts;p++){
                for(int g=0;g<local_group;g++){
                    int local_seq=static_cast<int>(grpo::idx2(p,g,local_group));
                    int global_g=device_id*local_group+g;
                    int global_seq=static_cast<int>(grpo::idx2(p,global_g,global_group));
                    auto& rollout=shard.rollouts[local_seq];
                    rollout=sample_rollout(old_logits,p,rng[device_id]);
                    global_rewards[global_seq]=rollout.reward;

                    for(int t=0;t<rollout.length;t++){
                        int i=local_seq*max_length+t;
                        int state=rollout.state[t];
                        int action=rollout.action[t];
                        shard.logp_old[i]=action_logp(old_logits,state,action);
                        shard.logp_ref[i]=action_logp(reference,state,action);
                        shard.mask[i]=1;
                    }
                }
            }
        }

        auto global_advantages=grpo::group_advantages_cpu(
            global_rewards,prompts,global_group,grpo::AdvantageMode::standardized
        );
        for(int device_id=0;device_id<num_devices;device_id++){
            for(int p=0;p<prompts;p++){
                for(int g=0;g<local_group;g++){
                    int local_seq=static_cast<int>(grpo::idx2(p,g,local_group));
                    int global_g=device_id*local_group+g;
                    int global_seq=static_cast<int>(grpo::idx2(p,global_g,global_group));
                    shards[device_id].advantages[local_seq]=global_advantages[global_seq];
                }
            }
        }
    }

    struct RunResult{
        float initial=0;
        float final=0;
        bool greedy=false;
        bool finite=true;
        bool matches_cpu=true;
        std::array<Shard,num_devices> shards;
    };

    RunResult train(unsigned seed){
        std::array<std::mt19937,num_devices> rng{
            std::mt19937(seed*2),std::mt19937(seed*2+1)
        };
        std::vector<float> logits(states*actions,0.0f);
        std::vector<float> reference=logits;
        RunResult run;
        for(int p=0;p<prompts;p++) run.initial+=exact_success(logits,p)/prompts;

        grpo::LossConfig config;
        config.clip_eps=0.2f;
        config.beta=0.01f;
        config.reduction=grpo::ReductionMode::sequence_mean;

        int local_sequences=prompts*local_group;
        int local_tokens=local_sequences*max_length;

        for(int update=0;update<outer_updates;update++){
            auto old_logits=logits;
            sample_group(old_logits,reference,rng,run.shards);

            for(int pass=0;pass<passes;pass++){
                std::array<std::vector<float>,num_devices> logp_new;
                for(int device_id=0;device_id<num_devices;device_id++){
                    logp_new[device_id].assign(local_tokens,0.0f);
                    for(int seq=0;seq<local_sequences;seq++){
                        const auto& rollout=run.shards[device_id].rollouts[seq];
                        for(int t=0;t<rollout.length;t++){
                            int i=seq*max_length+t;
                            logp_new[device_id][i]=action_logp(
                                logits,rollout.state[t],rollout.action[t]
                            );
                        }
                    }
                }

                std::array<grpo::LossResult,num_devices> results;
                std::array<std::thread,num_devices> workers;
                for(int device_id=0;device_id<num_devices;device_id++){
                    workers[device_id]=std::thread([&,device_id]{
                        CUDA_CHECK(cudaSetDevice(device_id));
                        results[device_id]=grpo::grpo_loss_cuda(
                            logp_new[device_id],
                            run.shards[device_id].logp_old,
                            run.shards[device_id].logp_ref,
                            run.shards[device_id].advantages,
                            run.shards[device_id].mask,
                            prompts,local_group,max_length,config
                        );
                    });
                }
                for(auto& worker:workers) worker.join();

                if(update==0 && pass==0){
                    int global_sequences=prompts*global_group;
                    int global_tokens=global_sequences*max_length;
                    std::vector<float> global_new(global_tokens,0.0f);
                    std::vector<float> global_old(global_tokens,0.0f);
                    std::vector<float> global_ref(global_tokens,0.0f);
                    std::vector<float> global_advantages(global_sequences,0.0f);
                    std::vector<int> global_mask(global_tokens,0);
                    for(int device_id=0;device_id<num_devices;device_id++){
                        for(int p=0;p<prompts;p++){
                            for(int g=0;g<local_group;g++){
                                int local_seq=static_cast<int>(grpo::idx2(p,g,local_group));
                                int global_g=device_id*local_group+g;
                                int global_seq=static_cast<int>(grpo::idx2(p,global_g,global_group));
                                global_advantages[global_seq]=run.shards[device_id].advantages[local_seq];
                                for(int t=0;t<max_length;t++){
                                    int local_i=local_seq*max_length+t;
                                    int global_i=global_seq*max_length+t;
                                    global_new[global_i]=logp_new[device_id][local_i];
                                    global_old[global_i]=run.shards[device_id].logp_old[local_i];
                                    global_ref[global_i]=run.shards[device_id].logp_ref[local_i];
                                    global_mask[global_i]=run.shards[device_id].mask[local_i];
                                }
                            }
                        }
                    }
                    auto cpu=grpo::grpo_loss_cpu(
                        global_new,global_old,global_ref,global_advantages,global_mask,
                        prompts,global_group,max_length,config
                    );
                    float sharded_loss=(results[0].stats.loss+results[1].stats.loss)/num_devices;
                    run.matches_cpu=run.matches_cpu && std::fabs(cpu.stats.loss-sharded_loss)<2e-5f;
                    for(int device_id=0;device_id<num_devices;device_id++){
                        for(int p=0;p<prompts;p++){
                            for(int g=0;g<local_group;g++){
                                int local_seq=static_cast<int>(grpo::idx2(p,g,local_group));
                                int global_g=device_id*local_group+g;
                                int global_seq=static_cast<int>(grpo::idx2(p,global_g,global_group));
                                for(int t=0;t<max_length;t++){
                                    int local_i=local_seq*max_length+t;
                                    int global_i=global_seq*max_length+t;
                                    float sharded_grad=results[device_id].dlogp_new[local_i]/num_devices;
                                    run.matches_cpu=run.matches_cpu &&
                                        std::fabs(cpu.dlogp_new[global_i]-sharded_grad)<2e-5f;
                                }
                            }
                        }
                    }
                }

                std::vector<float> gradient(logits.size(),0.0f);
                for(int device_id=0;device_id<num_devices;device_id++){
                    auto& shard=run.shards[device_id];
                    shard.loss_calls++;
                    shard.valid_tokens+=results[device_id].stats.valid_tokens;
                    bool nonzero=false;

                    for(int seq=0;seq<local_sequences;seq++){
                        const auto& rollout=shard.rollouts[seq];
                        for(int t=0;t<rollout.length;t++){
                            int i=seq*max_length+t;
                            int state=rollout.state[t];
                            int chosen=rollout.action[t];
                            auto p=probabilities(logits,state);
                            float token_grad=results[device_id].dlogp_new[i]/num_devices;
                            nonzero=nonzero || std::fabs(token_grad)>1e-12f;
                            for(int action=0;action<actions;action++){
                                float dlogp=(action==chosen ? 1.0f : 0.0f)-p[action];
                                gradient[state*actions+action]+=token_grad*dlogp;
                            }
                        }
                    }
                    if(nonzero) shard.nonzero_calls++;
                    run.finite=run.finite && std::isfinite(results[device_id].stats.loss);
                }

                for(size_t i=0;i<logits.size();i++){
                    logits[i]-=learning_rate*gradient[i];
                    run.finite=run.finite && std::isfinite(logits[i]) && std::isfinite(gradient[i]);
                }
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
    int device_count=0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if(device_count<num_devices){
        std::cout << "two-GPU check skipped: needs 2 CUDA devices, found "
                  << device_count << "\n";
        return 77;
    }

    for(int device_id=0;device_id<num_devices;device_id++){
        cudaDeviceProp device{};
        CUDA_CHECK(cudaGetDeviceProperties(&device,device_id));
        std::cout << "device " << device_id << ": " << device.name
                  << " (sm_" << device.major << device.minor << ")\n";
    }

    bool passed=true;
    std::array<float,3> finals{};
    std::cout << std::fixed << std::setprecision(6);
    for(unsigned seed=0;seed<finals.size();seed++){
        auto run=train(seed);
        finals[seed]=run.final;
        std::cout << "seed " << seed
                  << ": initial=" << run.initial
                  << " final=" << run.final
                  << " greedy=" << (run.greedy ? "yes" : "no")
                  << " finite=" << (run.finite ? "yes" : "no")
                  << " matches_cpu=" << (run.matches_cpu ? "yes" : "no") << "\n";
        for(int device_id=0;device_id<num_devices;device_id++){
            const auto& shard=run.shards[device_id];
            std::cout << "  device " << device_id
                      << " calls=" << shard.loss_calls
                      << " active_calls=" << shard.nonzero_calls
                      << " tokens=" << shard.valid_tokens << "\n";
            passed=passed && shard.loss_calls==outer_updates*passes;
            passed=passed && shard.nonzero_calls>0 && shard.valid_tokens>0;
        }
        passed=passed && std::fabs(run.initial-0.0625f)<1e-6f;
        passed=passed && run.final>0.90f && run.greedy && run.finite && run.matches_cpu;
    }
    std::sort(finals.begin(),finals.end());
    std::cout << "median final=" << finals[1] << "\n";
    passed=passed && finals[1]>0.95f;
    if(!passed){
        std::cerr << "two-GPU check failed\n";
        return 1;
    }
}
