#include "grpo/cuda_utils.cuh"
#include "grpo/grpo_loss.hpp"
#include "grpo/grpo_loss_cuda.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifndef GRPO_BUILD_CONFIG
#define GRPO_BUILD_CONFIG "unknown"
#endif

namespace {
    struct Input{
        int B=0,G=0,T=0;
        std::vector<float> now,old,ref,advantages;
        std::vector<int> mask;
    };

    struct LogitsInput{
        int B=0,G=0,T=0,V=0;
        std::vector<float> logits,old,ref,advantages;
        std::vector<int> selected,mask;
    };

    Input make_input(int B, int G, int T, bool partial){
        Input x;
        x.B=B; x.G=G; x.T=T;
        int n_sequences=B*G;
        int n_tokens=n_sequences*T;
        x.now.resize(n_tokens);
        x.old.resize(n_tokens);
        x.ref.resize(n_tokens);
        x.advantages.resize(n_sequences);
        x.mask.assign(n_tokens,0);

        for(int seq=0;seq<n_sequences;seq++){
            x.advantages[seq]=(seq%2==0 ? 0.4f : -0.7f)+0.03f*(seq%5);
            int length=T;
            if(partial) length=std::max(1,T/2+(seq*37)%std::max(1,T/5+1));
            for(int t=0;t<T;t++){
                int i=seq*T+t;
                x.old[i]=-2.0f-0.000001f*i;
                float shift=-0.4f+0.8f*static_cast<float>(i%17)/16.0f;
                x.now[i]=x.old[i]+shift;
                x.ref[i]=x.old[i]+0.03f*static_cast<float>((i%5)-2);
                x.mask[i]=t<length;
            }
        }
        return x;
    }

    LogitsInput make_logits_input(int B, int G, int T, int V, bool partial){
        LogitsInput x;
        x.B=B; x.G=G; x.T=T; x.V=V;
        int n_sequences=B*G;
        int n_tokens=n_sequences*T;
        x.logits.resize(static_cast<size_t>(n_tokens)*V);
        x.old.resize(n_tokens);
        x.ref.resize(n_tokens);
        x.advantages.resize(n_sequences);
        x.selected.resize(n_tokens);
        x.mask.assign(n_tokens,0);

        std::vector<float> base(V);
        double base_sum=0.0;
        for(int v=0;v<V;v++){
            base[v]=-0.001f*static_cast<float>(v%257);
            base_sum+=std::exp(static_cast<double>(base[v]));
        }
        double base_lse=std::log(base_sum);
        for(int seq=0;seq<n_sequences;seq++){
            x.advantages[seq]=(seq%2==0 ? 0.6f : -0.4f)+0.02f*(seq%3);
            int length=partial ? 1+(seq*5)%T : T;
            for(int t=0;t<T;t++){
                int row=seq*T+t;
                bool active=t<length;
                x.mask[row]=active;
                x.selected[row]=active ? (row*37+11)%V : -1;
                float row_shift=0.01f*static_cast<float>(row%7);
                float* logits=x.logits.data()+static_cast<size_t>(row)*V;
                for(int v=0;v<V;v++) logits[v]=base[v]+row_shift;
                if(active){
                    float now=static_cast<float>(base[x.selected[row]]-base_lse);
                    float ratio_shift=-0.25f+0.1f*static_cast<float>(row%6);
                    x.old[row]=now-ratio_shift;
                    x.ref[row]=now+0.01f*static_cast<float>((row%5)-2);
                }else{
                    x.old[row]=0.0f;
                    x.ref[row]=0.0f;
                }
            }
        }
        return x;
    }

    bool stats_close(float got, float want){
        if(!std::isfinite(got) || !std::isfinite(want)) return false;
        return std::fabs(got-want)<=5e-4f+5e-4f*std::fabs(want);
    }

    void compare(
        const grpo::LossResult& got,
        const grpo::LossResult& want,
        const char* name
    ){
        if(!stats_close(got.stats.loss,want.stats.loss) ||
           !stats_close(got.stats.pg_loss,want.stats.pg_loss) ||
           !stats_close(got.stats.kl,want.stats.kl)){
            std::cerr << name << " scalar parity failed\n";
            std::exit(1);
        }
        if(got.stats.valid_tokens!=want.stats.valid_tokens){
            std::cerr << name << " valid-token count failed\n";
            std::exit(1);
        }
        if(got.dlogp_new.size()!=want.dlogp_new.size()){
            std::cerr << name << " gradient size failed\n";
            std::exit(1);
        }
        float max_error=0;
        for(size_t i=0;i<got.dlogp_new.size();i++){
            if(!std::isfinite(got.dlogp_new[i]) || !std::isfinite(want.dlogp_new[i])){
                std::cerr << name << " non-finite gradient\n";
                std::exit(1);
            }
            max_error=std::max(max_error,std::fabs(got.dlogp_new[i]-want.dlogp_new[i]));
        }
        if(max_error>2e-5f){
            std::cerr << name << " gradient parity failed: " << max_error << "\n";
            std::exit(1);
        }
    }

    void compare_logits(
        const grpo::LogitsLossResult& got,
        const grpo::LogitsLossResult& want,
        int n_tokens,
        int V,
        const char* name
    ){
        if(!stats_close(got.stats.loss,want.stats.loss) ||
           !stats_close(got.stats.pg_loss,want.stats.pg_loss) ||
           !stats_close(got.stats.kl,want.stats.kl)){
            std::cerr << name << " scalar parity failed\n";
            std::exit(1);
        }
        if(got.stats.valid_tokens!=want.stats.valid_tokens ||
           got.dlogits.size()!=want.dlogits.size()){
            std::cerr << name << " output shape failed\n";
            std::exit(1);
        }
        float max_error=0.0f;
        for(int row=0;row<n_tokens;row++){
            float row_sum=0.0f;
            for(int v=0;v<V;v++){
                size_t i=static_cast<size_t>(row)*V+v;
                if(!std::isfinite(got.dlogits[i]) || !std::isfinite(want.dlogits[i])){
                    std::cerr << name << " non-finite gradient\n";
                    std::exit(1);
                }
                max_error=std::max(max_error,std::fabs(got.dlogits[i]-want.dlogits[i]));
                row_sum+=got.dlogits[i];
            }
            if(std::fabs(row_sum)>3e-5f){
                std::cerr << name << " gradient row sum failed: " << row_sum << "\n";
                std::exit(1);
            }
        }
        if(max_error>3e-5f){
            std::cerr << name << " gradient parity failed: " << max_error << "\n";
            std::exit(1);
        }
    }

    template<class F>
    bool throws(F fn){
        try{
            fn();
        }catch(const std::runtime_error&){
            return true;
        }
        return false;
    }

    void check_cuda(){
        std::vector<std::tuple<int,int,int>> shapes={
            {1,1,1},{1,1,255},{1,1,256},{1,1,257},{1,1,4099},{2,3,257},
            {64,16,1024}
        };
        std::array<grpo::ReductionMode,3> reductions={
            grpo::ReductionMode::sequence_mean,
            grpo::ReductionMode::token_mean,
            grpo::ReductionMode::length_weighted
        };

        for(size_t i=0;i<shapes.size();i++){
            auto [B,G,T]=shapes[i];
            auto input=make_input(B,G,T,i==5);
            grpo::LossConfig config;
            config.clip_eps=0.2f;
            config.beta=0.03f;
            config.reduction=reductions[i%reductions.size()];
            config.length_alpha=0.5f;
            auto cpu=grpo::grpo_loss_cpu(
                input.now,input.old,input.ref,input.advantages,input.mask,B,G,T,config
            );
            auto atomic=grpo::grpo_loss_cuda(
                input.now,input.old,input.ref,input.advantages,input.mask,B,G,T,config,
                grpo::CudaLossKernel::atomic
            );
            auto block=grpo::grpo_loss_cuda(
                input.now,input.old,input.ref,input.advantages,input.mask,B,G,T,config,
                grpo::CudaLossKernel::block_reduce
            );
            compare(atomic,cpu,"atomic");
            compare(block,cpu,"block");
        }

        grpo::LossConfig config;
        bool cpu_empty=throws([&]{
            grpo::grpo_loss_cpu({0},{0},{0},{0},{0},1,1,1,config);
        });
        bool cuda_empty=throws([&]{
            grpo::grpo_loss_cuda({0},{0},{0},{0},{0},1,1,1,config);
        });
        bool cpu_overflow=throws([&]{
            grpo::grpo_loss_cpu({0},{-100},{0},{-1},{1},1,1,1,config);
        });
        bool cuda_overflow=throws([&]{
            grpo::grpo_loss_cuda({0},{-100},{0},{-1},{1},1,1,1,config);
        });
        if(!cpu_empty || !cuda_empty || !cpu_overflow || !cuda_overflow){
            std::cerr << "CPU/CUDA error contract failed\n";
            std::exit(1);
        }

        auto cpu_zero=grpo::grpo_loss_cpu({0},{-100},{0},{0},{1},1,1,1,config);
        auto cuda_zero=grpo::grpo_loss_cuda({0},{-100},{0},{0},{1},1,1,1,config);
        compare(cuda_zero,cpu_zero,"zero advantage");
        std::cout << "CUDA checks passed (" << shapes.size() << " shapes)\n";
    }

    void check_logits_cuda(){
        std::vector<std::tuple<int,int,int,int,bool>> cases={
            {1,2,3,17,true},
            {2,2,2,257,false},
            {1,1,2,4099,false}
        };
        for(auto [B,G,T,V,partial]:cases){
            auto input=make_logits_input(B,G,T,V,partial);
            grpo::LossConfig config;
            config.beta=0.03f;
            auto cpu=grpo::grpo_logits_cpu(
                input.logits,input.old,input.ref,input.selected,input.advantages,input.mask,
                B,G,T,V,config
            );
            auto separate=grpo::grpo_logits_cuda(
                input.logits,input.old,input.ref,input.selected,input.advantages,input.mask,
                B,G,T,V,config,grpo::CudaLogitsKernel::separate
            );
            auto fused=grpo::grpo_logits_cuda(
                input.logits,input.old,input.ref,input.selected,input.advantages,input.mask,
                B,G,T,V,config,grpo::CudaLogitsKernel::fused
            );
            int n_tokens=B*G*T;
            compare_logits(separate,cpu,n_tokens,V,"separate logits");
            compare_logits(fused,cpu,n_tokens,V,"fused logits");

            float* device_logits=nullptr;
            size_t logits_bytes=input.logits.size()*sizeof(float);
            CUDA_CHECK(cudaMalloc((void**)&device_logits,logits_bytes));
            CUDA_CHECK(cudaMemcpy(
                device_logits,input.logits.data(),logits_bytes,cudaMemcpyHostToDevice
            ));
            auto device_stats=grpo::grpo_logits_cuda_device_inplace(
                device_logits,input.old,input.ref,input.selected,input.advantages,input.mask,
                B,G,T,V,V,config
            );
            grpo::LogitsLossResult in_place;
            in_place.stats=device_stats;
            in_place.dlogits.resize(input.logits.size());
            CUDA_CHECK(cudaMemcpy(
                in_place.dlogits.data(),device_logits,logits_bytes,cudaMemcpyDeviceToHost
            ));
            CUDA_CHECK(cudaFree(device_logits));
            compare_logits(in_place,cpu,n_tokens,V,"in-place logits");
        }

        {
            auto input=make_logits_input(1,2,3,17,true);
            constexpr int P=32;
            int n_tokens=input.B*input.G*input.T;
            std::vector<float> padded(static_cast<size_t>(n_tokens)*P,123.0f);
            for(int row=0;row<n_tokens;row++){
                std::copy_n(
                    input.logits.data()+static_cast<size_t>(row)*input.V,input.V,
                    padded.data()+static_cast<size_t>(row)*P
                );
            }
            grpo::LossConfig padded_config;
            padded_config.beta=0.03f;
            auto cpu=grpo::grpo_logits_cpu(
                input.logits,input.old,input.ref,input.selected,input.advantages,input.mask,
                input.B,input.G,input.T,input.V,padded_config
            );
            float* device_logits=nullptr;
            size_t padded_bytes=padded.size()*sizeof(float);
            CUDA_CHECK(cudaMalloc((void**)&device_logits,padded_bytes));
            CUDA_CHECK(cudaMemcpy(
                device_logits,padded.data(),padded_bytes,cudaMemcpyHostToDevice
            ));
            auto stats=grpo::grpo_logits_cuda_device_inplace(
                device_logits,input.old,input.ref,input.selected,input.advantages,input.mask,
                input.B,input.G,input.T,input.V,P,padded_config
            );
            CUDA_CHECK(cudaMemcpy(
                padded.data(),device_logits,padded_bytes,cudaMemcpyDeviceToHost
            ));
            CUDA_CHECK(cudaFree(device_logits));
            if(!stats_close(stats.loss,cpu.stats.loss) ||
               !stats_close(stats.pg_loss,cpu.stats.pg_loss) ||
               !stats_close(stats.kl,cpu.stats.kl)){
                std::cerr << "padded in-place scalar parity failed\n";
                std::exit(1);
            }
            for(int row=0;row<n_tokens;row++){
                for(int v=0;v<P;v++){
                    float got=padded[static_cast<size_t>(row)*P+v];
                    float want=v<input.V
                        ? cpu.dlogits[static_cast<size_t>(row)*input.V+v]
                        : 0.0f;
                    if(!std::isfinite(got) || std::fabs(got-want)>3e-5f){
                        std::cerr << "padded in-place gradient parity failed\n";
                        std::exit(1);
                    }
                }
            }
        }

        grpo::LossConfig config;
        auto zero=make_logits_input(1,1,1,3,false);
        zero.advantages[0]=0.0f;
        zero.old[0]=-101.0f;
        zero.ref[0]=zero.old[0];
        auto cpu=grpo::grpo_logits_cpu(
            zero.logits,zero.old,zero.ref,zero.selected,zero.advantages,zero.mask,
            1,1,1,3,config
        );
        auto fused=grpo::grpo_logits_cuda(
            zero.logits,zero.old,zero.ref,zero.selected,zero.advantages,zero.mask,
            1,1,1,3,config,grpo::CudaLogitsKernel::fused
        );
        compare_logits(fused,cpu,1,3,"zero-advantage logits");
        std::cout << "CUDA logits checks passed (" << cases.size()+1 << " cases)\n";
    }

    void print_machine(){
        cudaDeviceProp device;
        CUDA_CHECK(cudaGetDeviceProperties(&device,0));
        int runtime=0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        std::cout << "GPU: " << device.name << " (sm_" << device.major << device.minor << ")\n";
        std::cout << "CUDA runtime: " << runtime/1000 << "." << (runtime%1000)/10 << "\n";
        std::cout << "nvcc: " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__
                  << "." << __CUDACC_VER_BUILD__ << "\n";
        std::cout << "build: " << GRPO_BUILD_CONFIG << ", C++20\n\n";
    }

    void benchmark(){
        std::vector<std::tuple<int,int,int>> shapes={
            {8,8,128},{32,8,512},{64,16,1024},{64,16,4096}
        };
        std::cout << "Resident-device steady-state loss pass; includes accumulator reset "
                     "and kernel launch; excludes validation, weight construction, allocation, "
                     "transfers and result copies. Atomic is timed before block.\n";
        std::cout << "B G T mask active_% run active atomic_ms block_ms speedup block_Gtok/s\n";
        std::cout << std::fixed << std::setprecision(4);
        for(auto [B,G,T]:shapes){
            int n=B*G*T;
            int iterations=n<=131072 ? 1000 : 200;
            int runs=n>=1048576 ? 3 : 1;
            for(bool partial:{false,true}){
                auto input=make_input(B,G,T,partial);
                grpo::LossConfig config;
                config.clip_eps=0.2f;
                config.beta=0.01f;
                config.reduction=grpo::ReductionMode::token_mean;
                int active=std::count(input.mask.begin(),input.mask.end(),1);
                double active_percent=100.0*active/static_cast<double>(n);
                for(int run=1;run<=runs;run++){
                    auto timing=grpo::benchmark_grpo_loss_cuda(
                        input.now,input.old,input.ref,input.advantages,input.mask,
                        B,G,T,config,100,iterations
                    );
                    double throughput=active/(timing.block_ms*1e6);
                    std::cout << B << " " << G << " " << T << " "
                              << (partial ? "ragged" : "full") << " "
                              << std::setprecision(1) << active_percent << " "
                              << run << " " << std::setprecision(4)
                              << active << " " << timing.atomic_ms << " "
                              << timing.block_ms << " " << timing.atomic_ms/timing.block_ms
                              << " " << throughput << "\n";
                }
            }
        }
    }

    void benchmark_logits(){
        std::vector<std::tuple<int,int,int,int,bool>> cases={
            {32,8,4,4096,false},
            {32,8,4,32768,false},
            {32,8,4,32768,true},
            {32,8,4,131072,false},
            {256,8,4,4096,false},
            {1024,8,4,4096,false}
        };
        std::cout << "\nResident-device logits forward, GRPO, and full logits backward. "
                     "Includes accumulator reset and every GPU stage; excludes setup and transfers. "
                     "Timing order alternates across 15 samples.\n";
        std::cout << "tokens V mask active_% separate_ms fused_ms speedup\n";
        std::cout << std::fixed << std::setprecision(4);
        for(auto [B,G,T,V,partial]:cases){
            auto input=make_logits_input(B,G,T,V,partial);
            int n_tokens=B*G*T;
            int active=std::count(input.mask.begin(),input.mask.end(),1);
            size_t work=static_cast<size_t>(n_tokens)*V;
            int iterations=work<=8*1024*1024 ? 100 : (work<=64*1024*1024 ? 30 : 10);
            grpo::LossConfig config;
            config.beta=0.01f;
            auto timing=grpo::benchmark_grpo_logits_cuda(
                input.logits,input.old,input.ref,input.selected,input.advantages,input.mask,
                B,G,T,V,config,10,iterations
            );
            std::cout << n_tokens << " " << V << " "
                      << (partial ? "ragged" : "full") << " "
                      << std::setprecision(1) << 100.0*active/n_tokens << " "
                      << std::setprecision(4) << timing.separate_ms << " "
                      << timing.fused_ms << " " << timing.separate_ms/timing.fused_ms << "\n";
        }
    }
}

int main(int argc, char** argv){
    check_cuda();
    check_logits_cuda();
    if(argc==2 && std::string(argv[1])=="--check-only") return 0;
    if(argc==2 && std::string(argv[1])=="--logits-only"){
        print_machine();
        benchmark_logits();
        return 0;
    }
    if(argc!=1){
        std::cerr << "usage: bench_grpo_loss_cuda [--check-only|--logits-only]\n";
        return 2;
    }
    print_machine();
    benchmark();
    benchmark_logits();
}
