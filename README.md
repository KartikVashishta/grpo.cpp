# grpo.cpp

I wanted a GRPO implementation small enough to step through in a debugger.

This one trains a 24-logit autoregressive policy in plain C++: sample a group,
score it, compute relative advantages, run the clipped loss, backpropagate
through log-softmax, and update the policy with SGD. The CUDA path evaluates the
same token loss and can backpropagate from vocabulary logits. It does not hide
a transformer or an autograd engine.

```text
rollouts -> rewards -> group advantages -> clipped loss -> dlogp -> SGD
```

## Build

The CPU build has no CUDA dependency.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tiny_grpo_train
./build/length_tradeoff
```

For the CUDA loss and benchmarks:

```bash
cmake -S . -B build-cuda -DGRPO_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
./build-cuda/bench_grpo_loss_cuda
./build-cuda/bench_grpo_loss_cuda --logits-only
```

The GPT-2 example is kept out of the default build. It fetches one pinned
[`llm.c`](https://github.com/karpathy/llm.c) revision for the model code:

```bash
cmake -S . -B build-gpt2 \
  -DGRPO_USE_CUDA=ON -DGRPO_BUILD_GPT2=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpt2 -j
curl -L -o gpt2_124M.bin \
  https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_124M.bin
./build-gpt2/gpt2_grpo gpt2_124M.bin
```

Add `-DGRPO_BUILD_DISTRIBUTED=ON` if NCCL is installed and you want the
two-GPU example as well.

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

## CUDA

The naive kernel performs three global atomics for every active token. The
second kernel reduces inside each block, then performs three atomics per block.
Both are checked against the CPU result before the benchmark runs.

These are medians of three runs on a Tesla T4 with CUDA 12.8 and nvcc 12.8.93:

| tokens | mask | per-token atomics | block reduction |
| ---: | --- | ---: | ---: |
| 1,048,576 | full | 6.2334 ms | 0.1034 ms |
| 1,048,576 | ragged | 3.7385 ms | 0.0896 ms |
| 4,194,304 | full | 24.9207 ms | 0.4416 ms |
| 4,194,304 | ragged | 14.9457 ms | 0.3687 ms |

The arrays are already on the device for this measurement. A timed pass includes
the accumulator reset and kernel launch, but not allocation, validation,
weight construction, or transfers. The speedup is against the intentionally
simple atomic baseline, not an end-to-end training number.

There is a second benchmark at the model-facing boundary. The baseline runs
log-softmax and gather, the selected-token GRPO loss, and log-softmax backward
as three kernels. The fused path carries the row maximum and normalizer together
as in [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867),
then writes the full logits gradient without storing `logsumexp`, selected
log-probabilities, or `dlogp` between kernels.

These numbers are medians of 15 alternating-order samples on the same T4:

| tokens | vocabulary | mask | separate | fused |
| ---: | ---: | --- | ---: | ---: |
| 1,024 | 32,768 | full | 2.3959 ms | 1.8687 ms |
| 1,024 | 32,768 | ragged | 1.8158 ms | 1.4404 ms |
| 1,024 | 131,072 | full | 10.3036 ms | 7.9044 ms |

That is `1.26x` to `1.30x` for these rows. It is a fused loss-boundary result,
not a claim about transformer training throughput.

## A real model update

[`apps/04_gpt2_grpo.cu`](apps/04_gpt2_grpo.cu) runs the same loss against the
FP32 GPT-2 124M model from `llm.c`. Sampling is deliberately plain and still
comes back to the CPU. At the loss boundary the padded model logits stay in
their GPU buffer: the fused kernel turns them into `dL/dlogits` in place, then
`llm.c` carries that gradient through the transformer and AdamW.

The check uses eight fixed prompts, exact one-token rewards, a fixed initial
reference with `beta=0.01`, and four PPO passes per update. There are two
rollout allocators:

```bash
./build-gpt2/gpt2_grpo gpt2_124M.bin 4 uniform 42
./build-gpt2/gpt2_grpo gpt2_124M.bin 4 vigor 42
```

Uniform gives every prompt eight samples. The VIGOR path starts every prompt at
two samples, keeps the half with the highest cumulative reward variance, and
doubles the sampling count for four rounds. With eight prompts that produces
the sorted allocation `30 14 6 6 2 2 2 2`: still 64 samples in total, so the
comparison does not quietly buy more model output.

The allocator follows [VIGOR](https://arxiv.org/abs/2607.22002). On three seeded
four-update runs on an A100 PCIe 40 GB, the final mean target probabilities were:

| seed | uniform | VIGOR |
| ---: | ---: | ---: |
| 42 | 0.338416 | 0.339761 |
| 43 | 0.379996 | 0.307541 |
| 44 | 0.338991 | 0.393748 |
| mean | 0.352468 | 0.347017 |

That tiny check is a tie within the noise and does not demonstrate the paper's
sample-efficiency result. It does exercise the complete allocator and model
update under an equal rollout budget. A one-update VIGOR run under Compute
Sanitizer memcheck reported zero errors.

## Two-GPU GPT-2

[`apps/05_two_gpu_gpt2_grpo.cu`](apps/05_two_gpu_gpt2_grpo.cu) is the next step
past the earlier toy sharding example. It forks one process per GPU, gathers
rewards to compute advantages over the full 16-sample group, then averages the
GPT-2 parameter gradients with NCCL after every PPO pass.

```bash
cmake -S . -B build-gpt2 \
  -DGRPO_USE_CUDA=ON -DGRPO_BUILD_GPT2=ON \
  -DGRPO_BUILD_DISTRIBUTED=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpt2 -j
./build-gpt2/gpt2_grpo_2gpu gpt2_124M.bin 4
```

On two A100 PCIe 40 GB cards with CUDA 12.8 and NCCL 2.26.2, two four-update
runs moved both replicas from `0.0597304` to `0.675478` and `0.666268`. The
program also hashed every parameter after training; the two hashes matched in
both runs. This is real synchronous data-parallel model training, but
deliberately not a launcher, checkpoint manager, or cluster framework.

## Two T4s

[`apps/03_two_gpu_train.cu`](apps/03_two_gpu_train.cu) splits each 16-sample
group across two devices, launches one host thread per device, and adds the two
gradient shards before SGD. It is a small single-process data-parallel check,
not DDP or an NCCL runtime; sampling and the policy update are still on the CPU.

On Kaggle's two T4s, the three seeded runs finished at `0.973275`, `0.972558`,
and `0.971674`. The first sharded update matched one global CPU batch. The CUDA
suite passed 5/5, followed by Compute Sanitizer memcheck and racecheck.

## References

- [DeepSeekMath](https://arxiv.org/abs/2402.03300) introduced GRPO.
- [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) introduced Dr.GRPO and discusses response-length bias.
- [DAPO](https://arxiv.org/abs/2503.14476) uses the global token-level policy loss.
- [On the Impossibility of Unbiased and Length-Invariant Policy Optimization with Outcome Rewards](https://arxiv.org/abs/2607.23364) gives the alpha trade-off used above.
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) gives the one-pass normalizer used by the fused logits kernel.
- [VIGOR](https://arxiv.org/abs/2607.22002) gives the variance-guided rollout allocation used by the GPT-2 example.
