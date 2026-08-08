// A deliberately small two-process data-parallel check. Each process owns one
// llm.c model replica; NCCL averages its parameter gradients after every pass.
#define TESTING
#include "train_gpt2_fp32.cu"

#include "grpo/grpo_loss.hpp"
#include "grpo/grpo_loss_cuda.cuh"

#include <nccl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#define NCCL_CHECK(x) \
    do { \
        ncclResult_t result=(x); \
        if(result!=ncclSuccess){ \
            std::cerr << "NCCL error: " << ncclGetErrorString(result) << "\n"; \
            std::exit(1); \
        } \
    } while(0)

namespace {
    constexpr int world_size=2;
    constexpr int task_count=4;
    constexpr int local_G=8;
    constexpr int global_G=world_size*local_G;
    constexpr int T=8;
    constexpr int gpt2_eot=50256;

    struct Task{
        std::vector<int> tokens;
        int answer;
    };

    const std::array<Task,task_count> tasks={{
        {{464,3139,286,4881,318},6342},
        {{464,3124,286,262,6766,318},4171},
        {{464,6697,286,3024,318},4692},
        {{7571,5556,734,318},1440}
    }};

    struct Batch{
        std::vector<float> old_logp;
        std::vector<float> ref_logp;
        std::vector<float> advantages;
        std::vector<int> selected;
        std::vector<int> mask;
    };

    struct Result{
        double before;
        double after;
        std::uint64_t parameter_hash;
    };

    double row_logsumexp(const float* row, int V){
        float maximum=*std::max_element(row,row+V);
        double sum=0.0;
        for(int v=0;v<V;v++) sum+=std::exp(static_cast<double>(row[v]-maximum));
        return static_cast<double>(maximum)+std::log(sum);
    }

    int sample_row(const float* row, int V, std::mt19937& rng){
        double lse=row_logsumexp(row,V);
        std::uniform_real_distribution<double> uniform(0.0,1.0);
        double coin=uniform(rng),cumulative=0.0;
        for(int v=0;v<V;v++){
            cumulative+=std::exp(static_cast<double>(row[v])-lse);
            if(coin<cumulative) return v;
        }
        return V-1;
    }

    std::vector<int> make_inputs(){
        std::vector<int> inputs(task_count*local_G*T,gpt2_eot);
        for(int task=0;task<task_count;task++){
            for(int g=0;g<local_G;g++){
                int seq=task*local_G+g;
                std::copy(
                    tasks[task].tokens.begin(),tasks[task].tokens.end(),
                    inputs.begin()+static_cast<size_t>(seq)*T
                );
            }
        }
        return inputs;
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

    double target_mean(GPT2& model, std::vector<int>& inputs){
        gpt2_forward(&model,inputs.data(),NULL,task_count*local_G,T);
        auto logits=copy_logits(model);
        int V=model.config.vocab_size;
        int P=model.config.padded_vocab_size;
        double total=0.0;
        for(int task=0;task<task_count;task++){
            int position=static_cast<int>(tasks[task].tokens.size())-1;
            int token_index=(task*local_G)*T+position;
            const float* row=logits.data()+static_cast<size_t>(token_index)*P;
            total+=std::exp(
                static_cast<double>(row[tasks[task].answer])-row_logsumexp(row,V)
            );
        }
        return total/task_count;
    }

    Batch sample_batch(
        int rank,
        GPT2& model,
        std::vector<int>& inputs,
        const std::vector<float>& reference_logits,
        ncclComm_t comm,
        std::mt19937& rng
    ){
        int n_sequences=task_count*local_G;
        int n_tokens=n_sequences*T;
        int V=model.config.vocab_size;
        int P=model.config.padded_vocab_size;
        gpt2_forward(&model,inputs.data(),NULL,n_sequences,T);
        auto logits=copy_logits(model);

        Batch batch;
        batch.old_logp.assign(n_tokens,0.0f);
        batch.ref_logp.assign(n_tokens,0.0f);
        batch.selected.assign(n_tokens,-1);
        batch.mask.assign(n_tokens,0);
        std::vector<float> local_rewards(n_sequences,0.0f);
        for(int task=0;task<task_count;task++){
            int position=static_cast<int>(tasks[task].tokens.size())-1;
            for(int g=0;g<local_G;g++){
                int seq=task*local_G+g;
                int token_index=seq*T+position;
                const float* row=logits.data()+static_cast<size_t>(token_index)*P;
                const float* ref=reference_logits.data()+static_cast<size_t>(token_index)*P;
                double lse=row_logsumexp(row,V);
                double ref_lse=row_logsumexp(ref,V);
                int selected=sample_row(row,V,rng);
                batch.old_logp[token_index]=static_cast<float>(row[selected]-lse);
                batch.ref_logp[token_index]=static_cast<float>(ref[selected]-ref_lse);
                batch.selected[token_index]=selected;
                batch.mask[token_index]=1;
                local_rewards[seq]=selected==tasks[task].answer ? 1.0f : 0.0f;
            }
        }

        float *device_local=nullptr,*device_all=nullptr;
        cudaCheck(cudaMalloc((void**)&device_local,n_sequences*sizeof(float)));
        cudaCheck(cudaMalloc((void**)&device_all,world_size*n_sequences*sizeof(float)));
        cudaCheck(cudaMemcpy(
            device_local,local_rewards.data(),n_sequences*sizeof(float),cudaMemcpyHostToDevice
        ));
        NCCL_CHECK(ncclAllGather(
            device_local,device_all,n_sequences,ncclFloat,comm,0
        ));
        std::vector<float> gathered(world_size*n_sequences);
        cudaCheck(cudaMemcpy(
            gathered.data(),device_all,gathered.size()*sizeof(float),cudaMemcpyDeviceToHost
        ));
        cudaCheck(cudaFree(device_local));
        cudaCheck(cudaFree(device_all));

        std::vector<float> global_rewards(task_count*global_G);
        for(int source=0;source<world_size;source++){
            for(int task=0;task<task_count;task++){
                for(int g=0;g<local_G;g++){
                    global_rewards[task*global_G+source*local_G+g]
                        =gathered[source*n_sequences+task*local_G+g];
                }
            }
        }
        auto global_advantages=grpo::group_advantages_cpu(
            global_rewards,task_count,global_G,grpo::AdvantageMode::standardized
        );
        batch.advantages.resize(n_sequences);
        for(int task=0;task<task_count;task++){
            for(int g=0;g<local_G;g++){
                batch.advantages[task*local_G+g]
                    =global_advantages[task*global_G+rank*local_G+g];
            }
        }
        if(rank==0){
            std::cout << "global rewards:";
            for(int task=0;task<task_count;task++){
                int successes=0;
                for(int g=0;g<global_G;g++)
                    successes+=static_cast<int>(global_rewards[task*global_G+g]);
                std::cout << " " << successes << "/" << global_G;
            }
            std::cout << "\n";
        }
        return batch;
    }

    __global__ void scale_kernel(float* values, size_t count, float scale){
        size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        if(i<count) values[i]*=scale;
    }

    void setup_cuda(int rank, ncclUniqueId id, ncclComm_t& comm){
        cudaCheck(cudaSetDevice(rank));
        cublasCheck(cublasCreate(&cublas_handle));
        cudaDeviceProp properties;
        cudaCheck(cudaGetDeviceProperties(&properties,rank));
        bool tf32=properties.major>=8;
        cublas_compute_type=tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
        cublasCheck(cublasSetMathMode(
            cublas_handle,tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH
        ));
        NCCL_CHECK(ncclCommInitRank(&comm,world_size,id,rank));
        std::cout << "rank " << rank << ": " << properties.name << "\n";
    }

    Result train_rank(int rank, ncclUniqueId id, const char* checkpoint, int updates){
        ncclComm_t comm;
        setup_cuda(rank,id,comm);
        GPT2 model;
        gpt2_build_from_checkpoint(&model,checkpoint);
        auto inputs=make_inputs();
        std::mt19937 rng(42+1000*rank);
        double before=target_mean(model,inputs);
        auto reference_logits=copy_logits(model);

        grpo::LossConfig config;
        config.beta=0.01f;
        int optimizer_step=0;
        for(int update=0;update<updates;update++){
            auto batch=sample_batch(rank,model,inputs,reference_logits,comm,rng);
            for(int pass=0;pass<4;pass++){
                gpt2_forward(&model,inputs.data(),NULL,task_count*local_G,T);
                auto stats=grpo::grpo_logits_cuda_device_inplace(
                    model.acts.output,batch.old_logp,batch.ref_logp,batch.selected,
                    batch.advantages,batch.mask,task_count,local_G,T,
                    model.config.vocab_size,model.config.padded_vocab_size,config
                );
                gpt2_zero_grad(&model);
                model.mean_loss=stats.loss;
                gpt2_backward(&model);
                NCCL_CHECK(ncclAllReduce(
                    model.grads_memory,model.grads_memory,model.num_parameters,
                    ncclFloat,ncclSum,comm,0
                ));
                int threads=256;
                int blocks=static_cast<int>((model.num_parameters+threads-1)/threads);
                scale_kernel<<<blocks,threads>>>(
                    model.grads_memory,model.num_parameters,1.0f/world_size
                );
                cudaCheck(cudaGetLastError());
                gpt2_update(
                    &model,1.0e-5f,0.9f,0.999f,1e-8f,0.0f,++optimizer_step
                );
            }
        }
        cudaCheck(cudaDeviceSynchronize());
        double after=target_mean(model,inputs);
        std::vector<float> parameters(model.num_parameters);
        cudaCheck(cudaMemcpy(
            parameters.data(),model.params_memory,
            model.num_parameters*sizeof(float),cudaMemcpyDeviceToHost
        ));
        std::uint64_t parameter_hash=1469598103934665603ULL;
        for(float value:parameters){
            std::uint32_t bits;
            std::memcpy(&bits,&value,sizeof(bits));
            parameter_hash^=bits;
            parameter_hash*=1099511628211ULL;
        }

        gpt2_free(&model);
        NCCL_CHECK(ncclCommDestroy(comm));
        cublasCheck(cublasDestroy(cublas_handle));
        return {before,after,parameter_hash};
    }
}

int main(int argc, char** argv){
    if(argc<2 || argc>3){
        std::cerr << "usage: gpt2_grpo_2gpu gpt2_124M.bin [updates]\n";
        return 2;
    }
    if(!std::filesystem::exists(argv[1])){
        std::cerr << "checkpoint not found: " << argv[1] << "\n";
        return 2;
    }
    int updates=argc==3 ? std::stoi(argv[2]) : 2;
    if(updates<=0) throw std::runtime_error("updates must be positive");

    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));
    int pipes[world_size][2];
    pid_t children[world_size];
    for(int rank=0;rank<world_size;rank++){
        if(pipe(pipes[rank])!=0) throw std::runtime_error("pipe failed");
        pid_t child=fork();
        if(child<0) throw std::runtime_error("fork failed");
        if(child==0){
            close(pipes[rank][0]);
            Result result=train_rank(rank,id,argv[1],updates);
            ssize_t written=write(pipes[rank][1],&result,sizeof(result));
            close(pipes[rank][1]);
            std::cout.flush();
            std::fflush(stdout);
            std::_Exit(written==sizeof(result) ? 0 : 1);
        }
        close(pipes[rank][1]);
        children[rank]=child;
    }

    Result results[world_size];
    bool ok=true;
    for(int rank=0;rank<world_size;rank++){
        ssize_t got=read(pipes[rank][0],&results[rank],sizeof(Result));
        close(pipes[rank][0]);
        int status=0;
        waitpid(children[rank],&status,0);
        ok=ok && got==sizeof(Result) && WIFEXITED(status) && WEXITSTATUS(status)==0;
    }
    if(!ok){
        std::cerr << "a training rank failed\n";
        return 1;
    }

    for(int rank=0;rank<world_size;rank++){
        std::cout << "rank " << rank << " mean target probability: "
                  << results[rank].before << " -> " << results[rank].after << "\n";
    }
    bool parameters_match=results[0].parameter_hash==results[1].parameter_hash;
    std::cout << "replica parameter hash match: "
              << (parameters_match ? "yes" : "no") << "\n";
    if(!(results[0].after>results[0].before) ||
       !(results[1].after>results[1].before) || !parameters_match){
        std::cerr << "two-GPU training check failed\n";
        return 1;
    }
    return 0;
}
