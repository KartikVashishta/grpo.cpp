#include "grpo/grpo_loss_cuda.cuh"
#include "grpo/cuda_utils.cuh"

#include <climits>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace grpo {
    static void expect_size(const char* name, size_t got, size_t want){
        if(got!=want){
            throw std::runtime_error(
                std::string(name)+" has size "+std::to_string(got)+
                ", expected "+std::to_string(want)
            );
        }
    }

    static std::vector<float> make_weights(
        const std::vector<int>& mask,
        int B,
        int G,
        int T,
        ReductionMode reduction,
        float length_alpha,
        int& valid_tokens
    ){
        auto shape=detail::checked_shape(B,G,T);
        size_t n_seq=shape.sequences;
        size_t n_tokens=shape.tokens;
        std::vector<int> lengths(n_seq,0);
        std::vector<float> weights(n_seq,0.0f);

        valid_tokens=0;
        for(size_t i=0; i<n_tokens; i++){
            if(mask[i]==0) continue;
            lengths[i/static_cast<size_t>(T)]++;
            valid_tokens++;
        }

        for(size_t seq=0; seq<n_seq; seq++){
            if(lengths[seq]==0){
                throw std::runtime_error("every sequence needs at least one valid token");
            }
            if(reduction==ReductionMode::sequence_mean){
                weights[seq]=1.0f/(static_cast<float>(n_seq)*lengths[seq]);
            }else if(reduction==ReductionMode::token_mean){
                weights[seq]=1.0f/static_cast<float>(valid_tokens);
            }else{
                weights[seq]=std::pow(static_cast<float>(lengths[seq]),length_alpha-1.0f)
                    /static_cast<float>(n_seq);
            }
        }
        return weights;
    }

    __device__ void token_loss(
        float logp_new,
        float logp_old,
        float logp_ref,
        float advantage,
        float weight,
        float clip_eps,
        float beta,
        float& loss,
        float& pg_loss,
        float& kl,
        float& grad
    ){
        pg_loss=0.0f;
        float pg_grad=0.0f;
        if(advantage!=0.0f){
            float ratio=expf(logp_new-logp_old);
            float clipped_ratio=fminf(fmaxf(ratio,1.0f-clip_eps),1.0f+clip_eps);
            pg_loss=-fminf(ratio*advantage,clipped_ratio*advantage);

            // Match the CPU convention: the exact boundary uses the ratio branch.
            bool clipped_high=advantage>0.0f && ratio>1.0f+clip_eps;
            bool clipped_low=advantage<0.0f && ratio<1.0f-clip_eps;
            pg_grad=clipped_high || clipped_low ? 0.0f : -advantage*ratio;
        }

        float d=logp_ref-logp_new;
        float expm1_d=expm1f(d);
        kl=expm1_d-d;
        loss=pg_loss+beta*kl;

        float kl_grad=-beta*expm1_d;
        loss*=weight;
        pg_loss*=weight;
        kl*=weight;
        grad=weight*(pg_grad+kl_grad);
    }

    __global__ void grpo_loss_atomic_kernel(
        const float* logp_new,
        const float* logp_old,
        const float* logp_ref,
        const float* advantages,
        const float* weights,
        const int* mask,
        int n_tokens,
        int T,
        float clip_eps,
        float beta,
        float* sums,
        float* dlogp
    ){
        int i=blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=n_tokens) return;
        if(mask[i]==0){
            dlogp[i]=0.0f;
            return;
        }

        float loss,pg_loss,kl,grad;
        token_loss(
            logp_new[i],logp_old[i],logp_ref[i],advantages[i/T],weights[i/T],
            clip_eps,beta,loss,pg_loss,kl,grad
        );
        dlogp[i]=grad;
        atomicAdd(sums+0,loss);
        atomicAdd(sums+1,pg_loss);
        atomicAdd(sums+2,kl);
    }

    __global__ void grpo_loss_block_kernel(
        const float* logp_new,
        const float* logp_old,
        const float* logp_ref,
        const float* advantages,
        const float* weights,
        const int* mask,
        int n_tokens,
        int T,
        float clip_eps,
        float beta,
        float* sums,
        float* dlogp
    ){
        extern __shared__ float shared[];
        float* loss_sum=shared;
        float* pg_sum=shared+blockDim.x;
        float* kl_sum=shared+2*blockDim.x;

        int tid=threadIdx.x;
        int i=blockIdx.x*blockDim.x+tid;
        float loss=0.0f,pg_loss=0.0f,kl=0.0f,grad=0.0f;

        if(i<n_tokens && mask[i]!=0){
            token_loss(
                logp_new[i],logp_old[i],logp_ref[i],advantages[i/T],weights[i/T],
                clip_eps,beta,loss,pg_loss,kl,grad
            );
        }
        if(i<n_tokens) dlogp[i]=grad;

        loss_sum[tid]=loss;
        pg_sum[tid]=pg_loss;
        kl_sum[tid]=kl;
        __syncthreads();

        for(int stride=blockDim.x/2; stride>0; stride>>=1){
            if(tid<stride){
                loss_sum[tid]+=loss_sum[tid+stride];
                pg_sum[tid]+=pg_sum[tid+stride];
                kl_sum[tid]+=kl_sum[tid+stride];
            }
            __syncthreads();
        }

        if(tid==0){
            atomicAdd(sums+0,loss_sum[0]);
            atomicAdd(sums+1,pg_sum[0]);
            atomicAdd(sums+2,kl_sum[0]);
        }
    }

    struct DeviceData{
        float* logp_new=nullptr;
        float* logp_old=nullptr;
        float* logp_ref=nullptr;
        float* advantages=nullptr;
        float* weights=nullptr;
        int* mask=nullptr;
        float* sums=nullptr;
        float* dlogp=nullptr;
    };

    static DeviceData copy_to_device(
        const std::vector<float>& logp_new,
        const std::vector<float>& logp_old,
        const std::vector<float>& logp_ref,
        const std::vector<float>& advantages,
        const std::vector<float>& weights,
        const std::vector<int>& mask
    ){
        DeviceData d;
        size_t token_bytes=logp_new.size()*sizeof(float);
        size_t seq_bytes=advantages.size()*sizeof(float);
        size_t mask_bytes=mask.size()*sizeof(int);

        CUDA_CHECK(cudaMalloc((void**)&d.logp_new,token_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d.logp_old,token_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d.logp_ref,token_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d.advantages,seq_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d.weights,seq_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d.mask,mask_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d.sums,3*sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d.dlogp,token_bytes));

        CUDA_CHECK(cudaMemcpy(d.logp_new,logp_new.data(),token_bytes,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.logp_old,logp_old.data(),token_bytes,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.logp_ref,logp_ref.data(),token_bytes,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.advantages,advantages.data(),seq_bytes,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.weights,weights.data(),seq_bytes,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.mask,mask.data(),mask_bytes,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d.sums,0,3*sizeof(float)));
        return d;
    }

    static void free_device(DeviceData& d){
        CUDA_CHECK(cudaFree(d.logp_new));
        CUDA_CHECK(cudaFree(d.logp_old));
        CUDA_CHECK(cudaFree(d.logp_ref));
        CUDA_CHECK(cudaFree(d.advantages));
        CUDA_CHECK(cudaFree(d.weights));
        CUDA_CHECK(cudaFree(d.mask));
        CUDA_CHECK(cudaFree(d.sums));
        CUDA_CHECK(cudaFree(d.dlogp));
    }

    static void launch_loss(
        const DeviceData& d,
        int n_tokens,
        int T,
        LossConfig config,
        CudaLossKernel kernel
    ){
        int threads=256;
        int blocks=(n_tokens+threads-1)/threads;

        if(kernel==CudaLossKernel::atomic){
            grpo_loss_atomic_kernel<<<blocks,threads>>>(
                d.logp_new,d.logp_old,d.logp_ref,d.advantages,d.weights,d.mask,
                n_tokens,T,config.clip_eps,config.beta,d.sums,d.dlogp
            );
        }else{
            grpo_loss_block_kernel<<<blocks,threads,3*threads*sizeof(float)>>>(
                d.logp_new,d.logp_old,d.logp_ref,d.advantages,d.weights,d.mask,
                n_tokens,T,config.clip_eps,config.beta,d.sums,d.dlogp
            );
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static void validate_inputs(
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

        size_t n_seq=shape.sequences;
        size_t n_tokens=shape.tokens;
        expect_size("logp_new",logp_new.size(),n_tokens);
        expect_size("logp_old",logp_old.size(),n_tokens);
        expect_size("logp_ref",logp_ref.size(),n_tokens);
        expect_size("advantages",advantages.size(),n_seq);
        expect_size("mask",mask.size(),n_tokens);
        for(size_t i=0;i<n_tokens;i++){
            if(!std::isfinite(logp_new[i]) || !std::isfinite(logp_old[i]) || !std::isfinite(logp_ref[i]))
                throw std::runtime_error("log probabilities must be finite");
            if(mask[i]!=0 && mask[i]!=1) throw std::runtime_error("mask values must be 0 or 1");
        }
        for(float advantage:advantages){
            if(!std::isfinite(advantage)) throw std::runtime_error("advantages must be finite");
        }
    }

    LossResult grpo_loss_cuda(
        const std::vector<float>& logp_new,
        const std::vector<float>& logp_old,
        const std::vector<float>& logp_ref,
        const std::vector<float>& advantages,
        const std::vector<int>& mask,
        int B,
        int G,
        int T,
        LossConfig config,
        CudaLossKernel kernel
    ){
        validate_inputs(logp_new,logp_old,logp_ref,advantages,mask,B,G,T,config);
        int valid_tokens=0;
        auto weights=make_weights(mask,B,G,T,config.reduction,config.length_alpha,valid_tokens);
        int n_tokens=static_cast<int>(detail::checked_shape(B,G,T).tokens);

        LossResult result;
        result.stats.valid_tokens=valid_tokens;
        result.dlogp_new.resize(n_tokens,0.0f);
        auto d=copy_to_device(logp_new,logp_old,logp_ref,advantages,weights,mask);
        launch_loss(d,n_tokens,T,config,kernel);
        CUDA_CHECK(cudaDeviceSynchronize());

        float sums[3]={0.0f,0.0f,0.0f};
        CUDA_CHECK(cudaMemcpy(sums,d.sums,3*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            result.dlogp_new.data(),d.dlogp,n_tokens*sizeof(float),cudaMemcpyDeviceToHost
        ));
        free_device(d);

        result.stats.loss=sums[0];
        result.stats.pg_loss=sums[1];
        result.stats.kl=sums[2];
        if(!std::isfinite(result.stats.loss) || !std::isfinite(result.stats.pg_loss) || !std::isfinite(result.stats.kl))
            throw std::runtime_error("CUDA loss became non-finite");
        for(float grad:result.dlogp_new){
            if(!std::isfinite(grad)) throw std::runtime_error("CUDA gradient became non-finite");
        }
        return result;
    }

    static float time_kernel(
        const DeviceData& d,
        int n_tokens,
        int T,
        LossConfig config,
        CudaLossKernel kernel,
        int iterations
    ){
        cudaEvent_t start,stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        std::vector<float> samples;
        for(int repeat=0;repeat<7;repeat++){
            CUDA_CHECK(cudaEventRecord(start));
            for(int i=0;i<iterations;i++){
                CUDA_CHECK(cudaMemset(d.sums,0,3*sizeof(float)));
                launch_loss(d,n_tokens,T,config,kernel);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed=0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed,start,stop));
            samples.push_back(elapsed/static_cast<float>(iterations));
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        std::sort(samples.begin(),samples.end());
        return samples[samples.size()/2];
    }

    CudaTiming benchmark_grpo_loss_cuda(
        const std::vector<float>& logp_new,
        const std::vector<float>& logp_old,
        const std::vector<float>& logp_ref,
        const std::vector<float>& advantages,
        const std::vector<int>& mask,
        int B,
        int G,
        int T,
        LossConfig config,
        int warmup,
        int iterations
    ){
        validate_inputs(logp_new,logp_old,logp_ref,advantages,mask,B,G,T,config);
        if(warmup<0 || iterations<=0) throw std::runtime_error("invalid benchmark iteration count");

        int valid_tokens=0;
        auto weights=make_weights(mask,B,G,T,config.reduction,config.length_alpha,valid_tokens);
        int n_tokens=static_cast<int>(detail::checked_shape(B,G,T).tokens);
        auto d=copy_to_device(logp_new,logp_old,logp_ref,advantages,weights,mask);
        for(int i=0; i<warmup; i++){
            CUDA_CHECK(cudaMemset(d.sums,0,3*sizeof(float)));
            launch_loss(d,n_tokens,T,config,CudaLossKernel::atomic);
            CUDA_CHECK(cudaMemset(d.sums,0,3*sizeof(float)));
            launch_loss(d,n_tokens,T,config,CudaLossKernel::block_reduce);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CudaTiming timing;
        timing.atomic_ms=time_kernel(
            d,n_tokens,T,config,CudaLossKernel::atomic,iterations
        );
        timing.block_ms=time_kernel(
            d,n_tokens,T,config,CudaLossKernel::block_reduce,iterations
        );
        free_device(d);
        return timing;
    }
}
