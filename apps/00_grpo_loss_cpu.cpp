#include "grpo/grpo_loss.hpp"

#include <iostream>
#include <vector>

int main(){
    int B = 1;
    int G = 4; 
    int T = 5;

    std::vector<float> rewards = {0.0f,0.3f,1.0f,0.7f};

    auto advantages = grpo::group_advantages_cpu(rewards, B, G);
    std::cout << "sequence advantages\n";

    for(int g=0; g<G; g++){
        std::cout
            << " completion " << g
            << " reward=" << rewards[g]
            << " advantage=" << advantages[g]
            << "\n";
    }

    std::vector<float> logp_old(B*G*T), logp_new(B*G*T), logp_ref(B*G*T);
    std::vector<int> mask(B*G*T,1);

    for(int b=0; b<B; b++){
        for(int g=0; g<G; g++){
            for(int t=0; t<T; t++){
                int i = grpo::idx3(b,g,t,G,T);

                // random numbers to test
                logp_old[i]=-1.0f-0.01f*static_cast<float>(i);
                float shift = 0.0f;

                if(g==0) shift=0.04f;
                if(g==1) shift=-0.01f;
                if(g==2) shift=0.08f;
                if(g==3) shift=0.03f;

                logp_new[i]=logp_old[i]+shift;
                logp_ref[i] = logp_old[i];
            }
        }
    }
    // act like completion 1 ended early
    for(int t=3; t<T; t++){
        int i = grpo::idx3(0,1,t,G,T);
        mask[i]=0;
    }

    float clip_eps=0.2f;
    float beta=0.01f;

    auto stats=grpo::grpo_loss_cpu(logp_new,logp_old,logp_ref,advantages,mask,B,G,T,clip_eps,beta);

    std::cout << "loss=" << stats.loss << "\n";
    std::cout << "pg_loss=" << stats.pg_loss << "\n";
    std::cout << "kl=" << stats.kl << "\n";
    std::cout << "valid_tokens=" << stats.valid_tokens << "\n";
}
