#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace gpt2_example {
    struct Task{
        std::vector<int> tokens;
        int answer;
    };

    inline double row_logsumexp(const float* row, int V){
        float maximum=*std::max_element(row,row+V);
        double sum=0.0;
        for(int v=0;v<V;v++) sum+=std::exp(static_cast<double>(row[v]-maximum));
        return static_cast<double>(maximum)+std::log(sum);
    }

    inline int sample_row(const float* row, int V, std::mt19937& rng){
        double lse=row_logsumexp(row,V);
        std::uniform_real_distribution<double> uniform(0.0,1.0);
        double coin=uniform(rng),cumulative=0.0;
        for(int v=0;v<V;v++){
            cumulative+=std::exp(static_cast<double>(row[v])-lse);
            if(coin<cumulative) return v;
        }
        return V-1;
    }

    template<std::size_t N>
    std::vector<int> make_inputs(
        const std::array<Task,N>& tasks,
        int copies,
        int T,
        int eot
    ){
        std::vector<int> inputs(N*copies*T,eot);
        for(std::size_t task=0;task<N;task++){
            for(int copy=0;copy<copies;copy++){
                std::size_t seq=task*copies+copy;
                std::copy(
                    tasks[task].tokens.begin(),tasks[task].tokens.end(),
                    inputs.begin()+seq*T
                );
            }
        }
        return inputs;
    }

    inline std::vector<float> copy_logits(const GPT2& model){
        std::size_t count=static_cast<std::size_t>(model.batch_size)*model.seq_len
            *model.config.padded_vocab_size;
        std::vector<float> logits(count);
        cudaCheck(cudaMemcpy(
            logits.data(),model.acts.output,count*sizeof(float),cudaMemcpyDeviceToHost
        ));
        return logits;
    }
}
