# grpo.cpp

I wanted a GRPO implementation small enough to step through in a debugger.

This one trains a 24-logit autoregressive policy in plain C++: sample a group,
score it, compute relative advantages, run the clipped loss, backpropagate
through log-softmax, and update the policy with SGD. It does not hide a
transformer or an autograd engine.

```text
rollouts -> rewards -> group advantages -> clipped loss -> dlogp -> SGD
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tiny_grpo_train
./build/length_tradeoff
```

## The toy policy

There are three prompts, three ordinary tokens, and EOS. A rollout gets reward
one only for these two-token answers:

```text
<p0> -> a <eos>
<p1> -> b <eos>
<p2> -> c <eos>
```

Each update samples 16 answers per prompt and reuses them for four PPO passes.
The old policy is fixed during those passes; the reference policy stays at the
initial uniform distribution. Exact success starts at `1/16` and the five test
seeds finish at:

```text
0.971157  0.973650  0.973283  0.974296  0.972896
```

The full loop is in [`apps/01_tiny_grpo_train.cpp`](apps/01_tiny_grpo_train.cpp).

## The length-normalization detail

The part I found most useful is that "GRPO loss" does not identify one reduction
for variable-length responses. This repo keeps three choices separate:

| reduction | total weight of response `i` |
| --- | ---: |
| sequence mean | `1 / (B G)` |
| token mean | `L_i / sum_j L_j` |
| alpha weighting | `L_i^alpha / (B G)` |

Sequence mean is the default. Token mean is the reduction used by DAPO. The
alpha path follows the family `L^(alpha-1)` from [On the Impossibility of
Unbiased and Length-Invariant Policy Optimization with Outcome
Rewards](https://arxiv.org/abs/2607.23364): `alpha=0` is the sequence-normalized
end, while `alpha=1` removes the length normalization.

[`apps/02_length_tradeoff.cpp`](apps/02_length_tradeoff.cpp) checks that family
on a continue/EOS policy where every outcome can be enumerated. With 100,000
sampled groups:

| alpha | exact expected update | sampled |
| ---: | ---: | ---: |
| 0.0 | 0.09167075 | 0.09133101 |
| 0.5 | 0.14661711 | 0.14614544 |
| 1.0 | 0.28812000 | 0.28735213 |

Only the `alpha=1` row is the derivative of expected reward in this toy problem;
the other rows are the updates induced by their respective length weights.
