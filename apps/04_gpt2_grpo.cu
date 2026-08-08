// The GPT-2 forward/backward below comes from llm.c at the commit pinned in
// CMakeLists.txt. This file only supplies the rollout and GRPO training loop.
#define TESTING
#include "train_gpt2_fp32.cu"

#include "grpo/grpo_loss.hpp"
#include "grpo/grpo_loss_cuda.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
    constexpr int gpt2_eot=50256;

    struct Task{
        const char* prompt;
        std::vector<int> tokens;
        int answer;
    };

    const std::array<Task,3> tasks={{
        {"The capital of France is",{464,3139,286,4881,318},6342},
        {"The color of the sky is",{464,3124,286,262,6766,318},4171},
        {"The opposite of hot is",{464,6697,286,3024,318},4692}
    }};

    std::vector<int> make_inputs(int G, int T){
        int n_sequences=static_cast<int>(tasks.size())*G;
        std::vector<int> inputs(static_cast<size_t>(n_sequences)*T,gpt2_eot);
        for(int b=0;b<static_cast<int>(tasks.size());b++){
            for(int g=0;g<G;g++){
                int seq=b*G+g;
                std::copy(
                    tasks[b].tokens.begin(),tasks[b].tokens.end(),
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

    std::array<double,3> target_probabilities(
        GPT2& model,
        std::vector<int>& inputs,
        int G,
        int T
    ){
        gpt2_forward(&model,inputs.data(),NULL,static_cast<int>(tasks.size())*G,T);
        auto logits=copy_logits(model);
        int V=model.config.vocab_size;
        int P=model.config.padded_vocab_size;
        std::array<double,3> probabilities{};
        for(int b=0;b<static_cast<int>(tasks.size());b++){
            int seq=b*G;
            int position=static_cast<int>(tasks[b].tokens.size())-1;
            const float* row=logits.data()+(static_cast<size_t>(seq)*T+position)*P;
            probabilities[b]=std::exp(static_cast<double>(row[tasks[b].answer])-row_logsumexp(row,V));
        }
        return probabilities;
    }

    void print_probabilities(const char* label, const std::array<double,3>& probabilities){
        std::cout << label;
        for(double probability:probabilities) std::cout << " " << probability;
        std::cout << "\n";
    }

    struct RolloutBatch{
        std::vector<float> old_logp;
        std::vector<float> ref_logp;
        std::vector<float> rewards;
        std::vector<float> advantages;
        std::vector<int> selected;
        std::vector<int> mask;
        std::array<int,3> successes{};
    };

    RolloutBatch sample_batch(
        GPT2& model,
        std::vector<int>& inputs,
        const std::vector<float>& reference_logits,
        int G,
        int T,
        std::mt19937& rng
    ){
        int B=static_cast<int>(tasks.size());
        int n_sequences=B*G;
        int n_tokens=n_sequences*T;
        int V=model.config.vocab_size;
        int P=model.config.padded_vocab_size;
        gpt2_forward(&model,inputs.data(),NULL,n_sequences,T);
        auto logits=copy_logits(model);

        RolloutBatch batch;
        batch.old_logp.assign(n_tokens,0.0f);
        batch.ref_logp.assign(n_tokens,0.0f);
        batch.rewards.assign(n_sequences,0.0f);
        batch.selected.assign(n_tokens,-1);
        batch.mask.assign(n_tokens,0);
        for(int b=0;b<B;b++){
            int position=static_cast<int>(tasks[b].tokens.size())-1;
            for(int g=0;g<G;g++){
                int seq=b*G+g;
                int token_index=seq*T+position;
                const float* row=logits.data()+static_cast<size_t>(token_index)*P;
                const float* reference_row=reference_logits.data()
                    +static_cast<size_t>(token_index)*P;
                double lse=row_logsumexp(row,V);
                double reference_lse=row_logsumexp(reference_row,V);
                int selected=sample_row(row,V,rng);
                float logp=static_cast<float>(static_cast<double>(row[selected])-lse);
                batch.old_logp[token_index]=logp;
                batch.ref_logp[token_index]=static_cast<float>(
                    static_cast<double>(reference_row[selected])-reference_lse
                );
                batch.selected[token_index]=selected;
                batch.mask[token_index]=1;
                if(selected==tasks[b].answer){
                    batch.rewards[seq]=1.0f;
                    batch.successes[b]++;
                }
            }
        }
        batch.advantages=grpo::group_advantages_cpu(
            batch.rewards,B,G,grpo::AdvantageMode::standardized
        );
        return batch;
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
    if(argc<2 || argc>3){
        std::cerr << "usage: gpt2_grpo gpt2_124M.bin [updates]\n";
        return 2;
    }
    int updates=argc==3 ? std::stoi(argv[2]) : 4;
    if(updates<=0) throw std::runtime_error("updates must be positive");
    if(!std::filesystem::exists(argv[1])){
        std::cerr << "checkpoint not found: " << argv[1] << "\n";
        return 2;
    }

    constexpr int G=32;
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
    auto inputs=make_inputs(G,T);
    std::mt19937 rng(42);
    std::cout << std::fixed << std::setprecision(6);
    auto before=target_probabilities(model,inputs,G,T);
    auto reference_logits=copy_logits(model);
    print_probabilities("target probability before:",before);

    grpo::LossConfig config;
    config.clip_eps=0.2f;
    config.beta=0.01f;
    config.reduction=grpo::ReductionMode::sequence_mean;
    int optimizer_step=0;
    for(int update=0;update<updates;update++){
        auto batch=sample_batch(model,inputs,reference_logits,G,T,rng);
        std::cout << "update " << update+1 << " rewards:"
                  << " " << batch.successes[0] << "/" << G
                  << " " << batch.successes[1] << "/" << G
                  << " " << batch.successes[2] << "/" << G << "\n";

        for(int pass=0;pass<passes;pass++){
            gpt2_forward(&model,inputs.data(),NULL,static_cast<int>(tasks.size())*G,T);
            auto stats=grpo::grpo_logits_cuda_device_inplace(
                model.acts.output,batch.old_logp,batch.ref_logp,batch.selected,
                batch.advantages,batch.mask,static_cast<int>(tasks.size()),G,T,
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
        print_probabilities("target probability now:   ",target_probabilities(model,inputs,G,T));
    }

    auto after=target_probabilities(model,inputs,G,T);
    double before_mean=(before[0]+before[1]+before[2])/3.0;
    double after_mean=(after[0]+after[1]+after[2])/3.0;
    std::cout << "mean target probability: " << before_mean << " -> " << after_mean << "\n";

    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));
    if(!(after_mean>before_mean)){
        std::cerr << "training did not improve the exact-token targets\n";
        return 1;
    }
    return 0;
}
