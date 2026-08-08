// The GPT-2 forward/backward below comes from llm.c at the commit pinned in
// CMakeLists.txt. This file only supplies the rollout and GRPO training loop.
#define TESTING
#include "train_gpt2_fp32.cu"

#include "grpo/grpo_loss.hpp"
#include "grpo/grpo_loss_cuda.cuh"
#include "grpo/vigor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    constexpr int gpt2_eot=50256;
    constexpr int task_count=8;

    struct Task{
        const char* prompt;
        std::vector<int> tokens;
        int answer;
    };

    const std::array<Task,task_count> tasks={{
        {"The capital of France is",{464,3139,286,4881,318},6342},
        {"The color of the sky is",{464,3124,286,262,6766,318},4171},
        {"The opposite of hot is",{464,6697,286,3024,318},4692},
        {"Two plus two is",{7571,5556,734,318},1440},
        {"The first month is",{464,717,1227,318},3269},
        {"The largest planet is",{464,4387,5440,318},22721},
        {"The day after Monday is",{464,1110,706,3321,318},3431},
        {"Water freezes at",{19184,44389,379},6632}
    }};

    enum class Allocation{
        uniform,
        vigor
    };

    std::vector<int> make_eval_inputs(int G, int T){
        std::vector<int> inputs(static_cast<size_t>(task_count*G)*T,gpt2_eot);
        for(int task=0;task<task_count;task++){
            for(int g=0;g<G;g++){
                int seq=task*G+g;
                std::copy(
                    tasks[task].tokens.begin(),tasks[task].tokens.end(),
                    inputs.begin()+static_cast<size_t>(seq)*T
                );
            }
        }
        return inputs;
    }

    double row_logsumexp(const float* row, int V){
        float maximum=*std::max_element(row,row+V);
        double sum=0.0;
        for(int v=0;v<V;v++) sum+=std::exp(static_cast<double>(row[v]-maximum));
        return static_cast<double>(maximum)+std::log(sum);
    }

    int sample_row(const float* row, int V, std::mt19937& rng){
        double lse=row_logsumexp(row,V);
        std::uniform_real_distribution<double> uniform(0.0,1.0);
        double coin=uniform(rng);
        double cumulative=0.0;
        for(int v=0;v<V;v++){
            cumulative+=std::exp(static_cast<double>(row[v])-lse);
            if(coin<cumulative) return v;
        }
        return V-1;
    }

    std::vector<float> copy_logits(const GPT2& model){
        size_t count=static_cast<size_t>(model.batch_size)*model.seq_len
            *model.config.padded_vocab_size;
        std::vector<float> logits(count);
        cudaCheck(cudaMemcpy(
            logits.data(),model.acts.output,count*sizeof(float),cudaMemcpyDeviceToHost
        ));
        return logits;
    }

    using Probabilities=std::array<double,task_count>;

    Probabilities target_probabilities(
        GPT2& model,
        std::vector<int>& inputs,
        int G,
        int T
    ){
        gpt2_forward(&model,inputs.data(),NULL,task_count*G,T);
        auto logits=copy_logits(model);
        int V=model.config.vocab_size;
        int P=model.config.padded_vocab_size;
        Probabilities probabilities{};
        for(int task=0;task<task_count;task++){
            int seq=task*G;
            int position=static_cast<int>(tasks[task].tokens.size())-1;
            const float* row=logits.data()+(static_cast<size_t>(seq)*T+position)*P;
            probabilities[task]=std::exp(
                static_cast<double>(row[tasks[task].answer])-row_logsumexp(row,V)
            );
        }
        return probabilities;
    }

    void print_probabilities(const char* label, const Probabilities& probabilities){
        std::cout << label;
        for(double probability:probabilities) std::cout << " " << probability;
        std::cout << "\n";
    }

    struct Sample{
        int token;
        float old_logp;
        float ref_logp;
        float reward;
    };

    struct RolloutBatch{
        std::vector<int> inputs;
        std::vector<float> old_logp;
        std::vector<float> ref_logp;
        std::vector<float> advantages;
        std::vector<int> selected;
        std::vector<int> mask;
        std::array<int,task_count> counts{};
        std::array<int,task_count> successes{};
    };

    Sample draw_sample(
        int task,
        const std::vector<float>& logits,
        const std::vector<float>& reference_logits,
        int G,
        int T,
        int V,
        int P,
        std::mt19937& rng
    ){
        int position=static_cast<int>(tasks[task].tokens.size())-1;
        int token_index=(task*G)*T+position;
        const float* row=logits.data()+static_cast<size_t>(token_index)*P;
        const float* reference_row=reference_logits.data()+static_cast<size_t>(token_index)*P;
        double lse=row_logsumexp(row,V);
        double reference_lse=row_logsumexp(reference_row,V);
        int selected=sample_row(row,V,rng);
        return {
            selected,
            static_cast<float>(static_cast<double>(row[selected])-lse),
            static_cast<float>(static_cast<double>(reference_row[selected])-reference_lse),
            selected==tasks[task].answer ? 1.0f : 0.0f
        };
    }

    RolloutBatch flatten_samples(
        const std::array<std::vector<Sample>,task_count>& samples,
        int T
    ){
        int n_sequences=0;
        for(const auto& group:samples) n_sequences+=static_cast<int>(group.size());
        int n_tokens=n_sequences*T;

        RolloutBatch batch;
        batch.inputs.assign(n_tokens,gpt2_eot);
        batch.old_logp.assign(n_tokens,0.0f);
        batch.ref_logp.assign(n_tokens,0.0f);
        batch.advantages.assign(n_sequences,0.0f);
        batch.selected.assign(n_tokens,-1);
        batch.mask.assign(n_tokens,0);

        int seq=0;
        for(int task=0;task<task_count;task++){
            int G=static_cast<int>(samples[task].size());
            batch.counts[task]=G;
            std::vector<float> rewards(G);
            for(int g=0;g<G;g++) rewards[g]=samples[task][g].reward;
            auto advantages=grpo::group_advantages_cpu(
                rewards,1,G,grpo::AdvantageMode::standardized
            );

            int position=static_cast<int>(tasks[task].tokens.size())-1;
            for(int g=0;g<G;g++,seq++){
                std::copy(
                    tasks[task].tokens.begin(),tasks[task].tokens.end(),
                    batch.inputs.begin()+static_cast<size_t>(seq)*T
                );
                int token_index=seq*T+position;
                batch.old_logp[token_index]=samples[task][g].old_logp;
                batch.ref_logp[token_index]=samples[task][g].ref_logp;
                batch.advantages[seq]=advantages[g];
                batch.selected[token_index]=samples[task][g].token;
                batch.mask[token_index]=1;
                batch.successes[task]+=static_cast<int>(samples[task][g].reward);
            }
        }
        return batch;
    }

    RolloutBatch sample_batch(
        GPT2& model,
        std::vector<int>& eval_inputs,
        const std::vector<float>& reference_logits,
        int G,
        int T,
        Allocation allocation,
        std::mt19937& rng
    ){
        int V=model.config.vocab_size;
        int P=model.config.padded_vocab_size;
        gpt2_forward(&model,eval_inputs.data(),NULL,task_count*G,T);
        auto logits=copy_logits(model);
        std::array<std::vector<Sample>,task_count> samples;

        if(allocation==Allocation::uniform){
            for(int task=0;task<task_count;task++){
                for(int g=0;g<G;g++){
                    samples[task].push_back(draw_sample(
                        task,logits,reference_logits,G,T,V,P,rng
                    ));
                }
            }
        }else{
            grpo::VigorAllocator allocator(task_count);
            while(true){
                int count=allocator.rollouts_per_prompt();
                for(int task:allocator.active_prompts()){
                    std::vector<float> rewards;
                    rewards.reserve(count);
                    for(int i=0;i<count;i++){
                        auto sample=draw_sample(
                            task,logits,reference_logits,G,T,V,P,rng
                        );
                        rewards.push_back(sample.reward);
                        samples[task].push_back(sample);
                    }
                    allocator.observe(task,rewards);
                }
                if(!allocator.refine()) break;
            }
        }
        return flatten_samples(samples,T);
    }

    void setup_cuda(){
        cudaCheck(cudaSetDevice(0));
        cudaDeviceProp properties;
        cudaCheck(cudaGetDeviceProperties(&properties,0));
        cublasCheck(cublasCreate(&cublas_handle));
        bool tf32=properties.major>=8;
        cublas_compute_type=tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
        cublasCheck(cublasSetMathMode(
            cublas_handle,tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH
        ));
        std::cout << "GPU: " << properties.name << ", TF32 " << (tf32 ? "on" : "off") << "\n";
    }
}

int main(int argc, char** argv){
    if(argc<2 || argc>5){
        std::cerr << "usage: gpt2_grpo gpt2_124M.bin [updates] [uniform|vigor] [seed]\n";
        return 2;
    }
    int updates=argc>=3 ? std::stoi(argv[2]) : 4;
    std::string allocation_name=argc>=4 ? argv[3] : "uniform";
    int seed=argc==5 ? std::stoi(argv[4]) : 42;
    if(updates<=0) throw std::runtime_error("updates must be positive");
    Allocation allocation;
    if(allocation_name=="uniform") allocation=Allocation::uniform;
    else if(allocation_name=="vigor") allocation=Allocation::vigor;
    else throw std::runtime_error("allocation must be uniform or vigor");
    if(!std::filesystem::exists(argv[1])){
        std::cerr << "checkpoint not found: " << argv[1] << "\n";
        return 2;
    }

    constexpr int G=8;
    // llm.c's attention softmax reads rows as float4 values.
    constexpr int T=8;
    constexpr int passes=4;
    constexpr float learning_rate=1.0e-5f;
    setup_cuda();

    GPT2 model;
    gpt2_build_from_checkpoint(&model,argv[1]);
    if(model.config.vocab_size!=50257 || model.config.padded_vocab_size<model.config.vocab_size){
        throw std::runtime_error("this example expects the llm.c GPT-2 124M checkpoint");
    }
    auto eval_inputs=make_eval_inputs(G,T);
    std::mt19937 rng(seed);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "allocation: " << allocation_name << ", 64 rollouts/update\n";
    auto before=target_probabilities(model,eval_inputs,G,T);
    auto reference_logits=copy_logits(model);
    print_probabilities("target probability before:",before);

    grpo::LossConfig config;
    config.clip_eps=0.2f;
    config.beta=0.01f;
    config.reduction=grpo::ReductionMode::sequence_mean;
    int optimizer_step=0;
    for(int update=0;update<updates;update++){
        auto batch=sample_batch(
            model,eval_inputs,reference_logits,G,T,allocation,rng
        );
        std::cout << "update " << update+1 << " rollouts:";
        for(int count:batch.counts) std::cout << " " << count;
        std::cout << " rewards:";
        for(int task=0;task<task_count;task++){
            std::cout << " " << batch.successes[task] << "/" << batch.counts[task];
        }
        std::cout << "\n";

        int n_sequences=static_cast<int>(batch.advantages.size());
        for(int pass=0;pass<passes;pass++){
            gpt2_forward(&model,batch.inputs.data(),NULL,n_sequences,T);
            auto stats=grpo::grpo_logits_cuda_device_inplace(
                model.acts.output,batch.old_logp,batch.ref_logp,batch.selected,
                batch.advantages,batch.mask,1,n_sequences,T,
                model.config.vocab_size,model.config.padded_vocab_size,config
            );
            gpt2_zero_grad(&model);
            model.mean_loss=stats.loss;
            gpt2_backward(&model);
            gpt2_update(
                &model,learning_rate,0.9f,0.999f,1e-8f,0.0f,++optimizer_step
            );
        }
        cudaCheck(cudaDeviceSynchronize());
        print_probabilities("target probability now:   ",target_probabilities(model,eval_inputs,G,T));
    }

    auto after=target_probabilities(model,eval_inputs,G,T);
    double before_mean=0.0,after_mean=0.0;
    for(int task=0;task<task_count;task++){
        before_mean+=before[task];
        after_mean+=after[task];
    }
    before_mean/=task_count;
    after_mean/=task_count;
    std::cout << "mean target probability: " << before_mean << " -> " << after_mean << "\n";

    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    if(!(after_mean>before_mean)){
        std::cerr << "training did not improve the exact-token targets\n";
        return 1;
    }
    return 0;
}
