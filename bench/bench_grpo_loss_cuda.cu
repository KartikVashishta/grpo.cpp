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
        std::cout << "CUDA checks passed (" << shapes.size() << " shapes)\n";
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
}

int main(int argc, char** argv){
    check_cuda();
    if(argc==2 && std::string(argv[1])=="--check-only") return 0;
    if(argc!=1){
        std::cerr << "usage: bench_grpo_loss_cuda [--check-only]\n";
        return 2;
    }
    print_machine();
    benchmark();
}
