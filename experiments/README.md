# Experiments

[Countdown](countdown/README.md) trains Qwen3-0.6B on arithmetic expressions.
The two smaller experiments below look at response length and rollout allocation.

## Response length

[`length_weighting.cpp`](length_weighting.cpp) enumerates a small continue/EOS
policy, then checks the result with 100,000 sampled groups. It varies the
per-token weight `L^(alpha-1)` described in [On the Impossibility of Unbiased
and Length-Invariant Policy Optimization with Outcome
Rewards](https://arxiv.org/abs/2607.23364).

```bash
cmake -S . -B build-exp \
  -DGRPO_BUILD_EXPERIMENTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-exp -j
./build-exp/length_weighting
```

| alpha | exact expected update | sampled |
| ---: | ---: | ---: |
| 0.0 | 0.09167075 | 0.09133101 |
| 0.5 | 0.14661711 | 0.14614544 |
| 1.0 | 0.28812000 | 0.28735213 |

The sampled gradient is multiplied by `G / (G - 1)`. That removes the finite
group scaling introduced when the same group supplies both a reward and its
centered baseline. This is one toy policy, not a reproduction of the paper.

## Variance-guided rollouts

[`VigorAllocator`](../include/grpo/vigor.hpp) implements the schedule from
[VIGOR](https://arxiv.org/abs/2607.22002): begin with two rollouts per prompt,
retain the half with the highest cumulative reward variance, and double their
budget over four rounds.

The GPT-2 example exposes it as an alternative sampling mode:

```bash
./build-gpt2/gpt2_grpo gpt2_124M.bin 4 vigor 42
```

For eight prompts the sorted allocation is `30 14 6 6 2 2 2 2`, exactly 64
rollouts—the same total used by uniform sampling. Three A100 runs gave:

| seed | uniform | VIGOR |
| ---: | ---: | ---: |
| 42 | 0.338416 | 0.339761 |
| 43 | 0.379996 | 0.307541 |
| 44 | 0.338991 | 0.393748 |
| mean | 0.352468 | 0.347017 |

VIGOR did not improve the mean score across these three runs.
