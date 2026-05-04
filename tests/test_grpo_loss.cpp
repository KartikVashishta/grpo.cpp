#include <grpo/grpo_loss.hpp>
#include <iostream>
#include <cmath>

static void close(float got, float want, float tol, const char* name){
    if (std::fabs(got - want) > tol) {
        std::cerr
            << name
            << " mismatch. got="
            << got
            << " want="
            << want
            << " tol="
            << tol
            << "\n";

        std::exit(1);
    }
}

int main(){
    {
        int B = 1;
        int G = 4;

        std::vector<float> rewards = {0.0f, 1.0f, 1.0f, 0.0f};
        auto adv = grpo::group_advantages_cpu(rewards,B,G);
        close(adv[0], -1.0f, 1e-4f, "adv[0]");
        close(adv[1],  1.0f, 1e-4f, "adv[1]");
        close(adv[2],  1.0f, 1e-4f, "adv[2]");
        close(adv[3], -1.0f, 1e-4f, "adv[3]");
    }

    {
        int B = 1;
        int G = 2;
        int T = 1;

        std::vector<float> rewards = {0.0f,1.0f};
        auto adv = grpo::group_advantages_cpu(rewards, B, G, 1e-12f);
        
        std::vector<float> logp_old = {0.0f,0.0f};
        std::vector<float> logp_new = {0.1f,0.2f};
        std::vector<float> logp_ref = logp_new;
        std::vector<int> mask = {1,1};

        float clip_eps = 0.5f;
        float beta = 0.0f;

        auto stats = grpo::grpo_loss_cpu(logp_new, logp_old, logp_ref, adv, mask, B, G, T, clip_eps, beta);
        
        float expected_surrogate = (-std::exp(0.1f) + std::exp(0.2f)) / 2.0f;
        float expected_loss = -expected_surrogate;

        close(stats.loss, expected_loss, 1e-5f, "loss");
        close(stats.kl, 0.0f, 1e-6f, "kl");

        if (stats.valid_tokens != 2) {
            std::cerr << "valid_tokens mismatch\n";
            return 1;
        }
    }

    std::cout << "test_grpo_loss passed\n";
}