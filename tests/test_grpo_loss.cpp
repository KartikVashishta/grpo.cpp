#include "grpo/grpo_loss.hpp"
#include "grpo/vigor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

static void close(float got, float want, float abs_tol, float rel_tol, const char* name){
    float tol=abs_tol+rel_tol*std::fabs(want);
    if(std::fabs(got-want)<=tol) return;
    std::cerr << name << " mismatch: got " << got << ", want " << want << "\n";
    std::exit(1);
}

static void check(bool ok, const char* name){
    if(ok) return;
    std::cerr << name << " failed\n";
    std::exit(1);
}

template<class F>
static void must_throw(F fn, const char* name){
    try{
        fn();
    }catch(const std::runtime_error&){
        return;
    }
    std::cerr << name << " did not throw\n";
    std::exit(1);
}

static grpo::LossResult loss(
    const std::vector<float>& now,
    const std::vector<float>& old,
    const std::vector<float>& ref,
    const std::vector<float>& advantages,
    const std::vector<int>& mask,
    int B,
    int G,
    int T,
    grpo::LossConfig config={}
){
    return grpo::grpo_loss_cpu(now,old,ref,advantages,mask,B,G,T,config);
}

static void test_advantages(){
    std::vector<float> rewards={0,1,1,0, 2,2,2,2};
    auto z=grpo::group_advantages_cpu(
        rewards,2,4,grpo::AdvantageMode::standardized,1e-12f
    );
    auto centered=grpo::group_advantages_cpu(
        rewards,2,4,grpo::AdvantageMode::centered
    );

    close(z[0],-1,1e-5f,0,"standardized advantage");
    close(z[1], 1,1e-5f,0,"standardized advantage");
    for(int i=4;i<8;i++) close(z[i],0,0,0,"equal reward advantage");
    auto zero_eps=grpo::group_advantages_cpu(
        {2,2,2,2},1,4,grpo::AdvantageMode::standardized,0
    );
    for(float value:zero_eps) close(value,0,0,0,"zero variance advantage");
    auto single=grpo::group_advantages_cpu(
        {2},1,1,grpo::AdvantageMode::standardized,0
    );
    close(single[0],0,0,0,"single reward advantage");

    std::vector<float> decimal_rewards(7,0.1f);
    auto decimal_z=grpo::group_advantages_cpu(
        decimal_rewards,1,7,grpo::AdvantageMode::standardized,0
    );
    auto decimal_centered=grpo::group_advantages_cpu(
        decimal_rewards,1,7,grpo::AdvantageMode::centered
    );
    for(float value:decimal_z) close(value,0,0,0,"equal decimal advantage");
    for(float value:decimal_centered) close(value,0,0,0,"equal decimal centered advantage");

    for(int b=0;b<2;b++){
        float mean=0;
        for(int g=0;g<4;g++) mean+=centered[grpo::idx2(b,g,4)];
        close(mean,0,1e-6f,0,"centered group sum");
    }

    float mean=0,var=0;
    for(int g=0;g<4;g++) mean+=z[g];
    mean/=4;
    for(int g=0;g<4;g++) var+=(z[g]-mean)*(z[g]-mean);
    var/=4;
    close(mean,0,1e-6f,0,"standardized mean");
    close(var,1,2e-5f,0,"standardized variance");
}

static void test_clipping(){
    std::vector<float> old(4,0);
    std::vector<float> now={std::log(1.1f),std::log(1.5f),std::log(0.9f),std::log(0.5f)};
    std::vector<float> ref=now;
    std::vector<float> advantages={1,1,-1,-1};
    std::vector<int> mask(4,1);
    auto result=loss(now,old,ref,advantages,mask,1,4,1);

    close(result.dlogp_new[0],-1.1f/4,1e-6f,0,"positive inside clip");
    close(result.dlogp_new[1],0,0,0,"positive above clip");
    close(result.dlogp_new[2],0.9f/4,1e-6f,0,"negative inside clip");
    close(result.dlogp_new[3],0,0,0,"negative below clip");
}

static void test_kl(){
    std::vector<float> now={0};
    std::vector<float> old={0};
    std::vector<float> ref={std::log(2.0f)};
    std::vector<float> advantages={0};
    std::vector<int> mask={1};
    grpo::LossConfig config;
    config.beta=0.3f;
    auto result=loss(now,old,ref,advantages,mask,1,1,1,config);

    close(result.stats.kl,1.0f-std::log(2.0f),1e-6f,0,"sampled KL");
    close(result.stats.loss,0.3f*(1.0f-std::log(2.0f)),1e-6f,0,"KL loss");
    close(result.dlogp_new[0],-0.3f,1e-6f,0,"KL derivative");

    config.beta=1;
    for(float d:{1e-3f,1e-4f,1e-6f}){
        ref[0]=d;
        result=loss(now,old,ref,advantages,mask,1,1,1,config);
        double exact_kl=std::expm1(static_cast<double>(d))-static_cast<double>(d);
        double exact_grad=-std::expm1(static_cast<double>(d));
        close(result.stats.kl,static_cast<float>(exact_kl),2e-14f,0.15f,"near-zero KL");
        close(result.dlogp_new[0],static_cast<float>(exact_grad),1e-10f,2e-6f,"near-zero KL derivative");
        check(result.stats.kl>=0.0f,"near-zero KL is non-negative");
    }
}

static void test_reductions(){
    std::vector<float> logp(8,0);
    std::vector<float> advantages={1,1};
    std::vector<int> mask={1,0,0,0, 1,1,1,1};
    grpo::LossConfig config;

    config.reduction=grpo::ReductionMode::sequence_mean;
    auto sequence=loss(logp,logp,logp,advantages,mask,1,2,4,config);
    close(-sequence.dlogp_new[0],0.5f,1e-6f,0,"sequence short mass");
    float long_mass=0;
    for(int i=4;i<8;i++) long_mass-=sequence.dlogp_new[i];
    close(long_mass,0.5f,1e-6f,0,"sequence long mass");

    config.reduction=grpo::ReductionMode::token_mean;
    auto token=loss(logp,logp,logp,advantages,mask,1,2,4,config);
    close(-token.dlogp_new[0],0.2f,1e-6f,0,"token short mass");
    long_mass=0;
    for(int i=4;i<8;i++) long_mass-=token.dlogp_new[i];
    close(long_mass,0.8f,1e-6f,0,"token long mass");

    config.reduction=grpo::ReductionMode::length_weighted;
    for(float alpha:{0.0f,0.5f,1.0f}){
        config.length_alpha=alpha;
        auto weighted=loss(logp,logp,logp,advantages,mask,1,2,4,config);
        float short_mass=-weighted.dlogp_new[0];
        long_mass=0;
        for(int i=4;i<8;i++) long_mass-=weighted.dlogp_new[i];
        close(short_mass,0.5f,1e-6f,0,"weighted short mass");
        close(long_mass,std::pow(4.0f,alpha)/2.0f,1e-6f,0,"weighted long mass");
    }
}

static void test_finite_differences(){
    int B=2,G=3,T=4;
    int n=B*G*T;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> base(-2.0f,-0.5f);
    std::uniform_real_distribution<float> shift(-0.12f,0.12f);
    std::vector<float> old(n),now(n),ref(n);
    std::vector<float> advantages={0.7f,-0.2f,-0.5f,-0.8f,0.3f,0.5f};
    std::vector<int> mask(n,0);

    for(int seq=0;seq<B*G;seq++){
        int length=1+seq%T;
        for(int t=0;t<T;t++){
            int i=seq*T+t;
            old[i]=base(rng);
            now[i]=old[i]+shift(rng);
            ref[i]=old[i]+shift(rng);
            mask[i]=t<length;
        }
    }

    grpo::LossConfig config;
    config.beta=0.03f;
    config.reduction=grpo::ReductionMode::length_weighted;
    config.length_alpha=0.5f;
    auto result=loss(now,old,ref,advantages,mask,B,G,T,config);
    float h=1e-3f;

    for(int i=0;i<n;i++){
        if(mask[i]==0) continue;
        auto plus=now,minus=now;
        plus[i]+=h;
        minus[i]-=h;
        float numeric=(
            loss(plus,old,ref,advantages,mask,B,G,T,config).stats.loss-
            loss(minus,old,ref,advantages,mask,B,G,T,config).stats.loss
        )/(2*h);
        close(result.dlogp_new[i],numeric,2e-4f,3e-3f,"finite difference");
    }
}

static void test_bad_inputs(){
    std::vector<float> logp={0,0};
    std::vector<float> advantages={1,-1};
    std::vector<int> mask={1,1};

    must_throw([&]{ loss(logp,logp,logp,advantages,mask,0,2,1); },"bad dimensions");
    must_throw([&]{ loss({0},logp,logp,advantages,mask,1,2,1); },"wrong vector length");
    must_throw([&]{
        auto config=grpo::LossConfig{}; config.clip_eps=-0.1f;
        loss(logp,logp,logp,advantages,mask,1,2,1,config);
    },"negative clip");
    must_throw([&]{
        auto config=grpo::LossConfig{}; config.beta=-0.1f;
        loss(logp,logp,logp,advantages,mask,1,2,1,config);
    },"negative beta");
    must_throw([&]{ loss(logp,logp,logp,advantages,{1,0},1,2,1); },"empty sequence");
    must_throw([&]{
        grpo::group_advantages_cpu({},65536,65536);
    },"shape overflow");
    must_throw([&]{
        auto config=grpo::LossConfig{}; config.length_alpha=1.1f;
        config.reduction=grpo::ReductionMode::length_weighted;
        loss(logp,logp,logp,advantages,mask,1,2,1,config);
    },"alpha outside study range");
    must_throw([&]{
        auto bad=logp; bad[0]=std::numeric_limits<float>::infinity();
        loss(bad,logp,logp,advantages,mask,1,2,1);
    },"non-finite input");
    must_throw([&]{
        loss({0},{-100},{0},{-1},{1},1,1,1);
    },"non-finite result");
}

static void test_zero_advantage(){
    auto result=loss({0},{-100},{0},{0},{1},1,1,1);
    close(result.stats.loss,0,0,0,"zero advantage loss");
    close(result.dlogp_new[0],0,0,0,"zero advantage gradient");
}

static float selected_logp(const std::vector<float>& logits, int row, int V, int selected){
    float max_logit=logits[row*V];
    for(int v=1;v<V;v++) max_logit=std::max(max_logit,logits[row*V+v]);
    double sum=0;
    for(int v=0;v<V;v++) sum+=std::exp(static_cast<double>(logits[row*V+v]-max_logit));
    return logits[row*V+selected]-max_logit-static_cast<float>(std::log(sum));
}

static void test_logits_boundary(){
    int B=1,G=2,T=2,V=4;
    std::vector<float> logits={
        1.0f,-0.5f,0.2f,0.8f,
        0.3f,0.1f,-0.4f,0.7f,
        -0.2f,0.5f,1.1f,0.0f,
        1000.0f,999.0f,998.0f,997.0f
    };
    std::vector<int> selected={3,0,2,-1};
    std::vector<int> mask={1,1,1,0};
    std::vector<float> now(4,0.0f);
    for(int i=0;i<3;i++) now[i]=selected_logp(logits,i,V,selected[i]);
    std::vector<float> old={now[0]-0.1f,now[1]+0.05f,now[2]-0.03f,0};
    std::vector<float> ref={now[0]+0.02f,now[1]-0.01f,now[2]+0.04f,0};
    std::vector<float> advantages={0.7f,-0.4f};
    grpo::LossConfig config;
    config.beta=0.03f;

    auto base=loss(now,old,ref,advantages,mask,B,G,T,config);
    auto result=grpo::grpo_logits_cpu(
        logits,old,ref,selected,advantages,mask,B,G,T,V,config
    );
    close(result.stats.loss,base.stats.loss,1e-7f,0,"logits loss");
    close(result.stats.pg_loss,base.stats.pg_loss,1e-7f,0,"logits policy loss");
    close(result.stats.kl,base.stats.kl,1e-7f,0,"logits KL");

    for(int i=0;i<4;i++){
        float row_sum=0;
        for(int v=0;v<V;v++) row_sum+=result.dlogits[i*V+v];
        close(row_sum,0,2e-7f,0,"logits gradient row sum");
    }
    for(int v=0;v<V;v++) close(result.dlogits[3*V+v],0,0,0,"masked logits gradient");

    float h=1e-3f;
    for(int i=0;i<3;i++){
        for(int v=0;v<V;v++){
            auto plus=logits,minus=logits;
            plus[i*V+v]+=h;
            minus[i*V+v]-=h;
            float upper=grpo::grpo_logits_cpu(
                plus,old,ref,selected,advantages,mask,B,G,T,V,config
            ).stats.loss;
            float lower=grpo::grpo_logits_cpu(
                minus,old,ref,selected,advantages,mask,B,G,T,V,config
            ).stats.loss;
            close(result.dlogits[i*V+v],(upper-lower)/(2*h),2e-5f,4e-3f,"logits finite difference");
        }
    }

    must_throw([&]{
        grpo::grpo_logits_cpu(logits,old,ref,{4,0,2,-1},advantages,mask,B,G,T,V,config);
    },"selected token outside vocabulary");
    must_throw([&]{
        auto bad=logits; bad[0]=std::numeric_limits<float>::infinity();
        grpo::grpo_logits_cpu(bad,old,ref,selected,advantages,mask,B,G,T,V,config);
    },"non-finite logits");
}

static void test_vigor_allocation(){
    grpo::VigorAllocator allocator(8);
    std::vector<std::vector<float>> streams(8,std::vector<float>(30,0.0f));
    for(int i=0;i<30;i++) streams[0][i]=static_cast<float>(i%2);
    streams[1][1]=1.0f;
    for(int i=4;i<30;i+=5) streams[1][i]=1.0f;
    streams[2][1]=1.0f;
    streams[3][1]=1.0f;

    while(true){
        int count=allocator.rollouts_per_prompt();
        for(int prompt:allocator.active_prompts()){
            int offset=allocator.rollout_count(prompt);
            std::vector<float> rewards(
                streams[prompt].begin()+offset,
                streams[prompt].begin()+offset+count
            );
            allocator.observe(prompt,rewards);
        }
        if(!allocator.refine()) break;
    }

    std::vector<int> want={30,14,6,6,2,2,2,2};
    int total=0;
    for(int prompt=0;prompt<8;prompt++){
        check(allocator.rollout_count(prompt)==want[prompt],"VIGOR rollout allocation");
        total+=allocator.rollout_count(prompt);
    }
    check(total==64,"VIGOR matched rollout budget");
    must_throw([&]{
        allocator.observe(7,std::vector<float>(16,0.0f));
    },"inactive VIGOR prompt");
}

int main(){
    test_advantages();
    test_clipping();
    test_kl();
    test_reductions();
    test_finite_differences();
    test_bad_inputs();
    test_zero_advantage();
    test_logits_boundary();
    test_vigor_allocation();
    std::cout << "test_grpo_loss passed\n";
}
