# grpo.cpp

I wanted to see a GRPO update without a Python trainer around it.

The toy example is a 24-logit policy you can follow from sampling to SGD.
There are also GPT-2 124M and Qwen3-0.6B examples. Qwen trains on three-number
Countdown: generate an expression, check the arithmetic, update the model.

On a 24 GB A10, two native runs took 143 and 131 seconds, about five cents of
estimated compute each. They scored 198/512 and 179/512 on the held-out test.
TRL took 439 and 392 seconds and scored 183/512 and 199/512. Native was faster
in both seeds, but the scores went either way. Costs exclude builds, startup,
storage, network and the separate final test.

The [Countdown experiment](experiments/countdown/README.md) has the run
commands, checkpoints, raw results and an answer viewer. Qwen's model and
kernels come from a pinned revision of [RL.cu](https://github.com/KJLdefeated/RL.cu/tree/77ecd8385d9836bc8377e042f4e2b7dd0b6df9c6).

![Countdown learning on A10](experiments/countdown/learning-a10.svg)

## Build

The default build is plain C++20:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tiny_grpo_train
```

CUDA adds the loss kernels and their benchmark:

```bash
cmake -S . -B build-cuda -DGRPO_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
./build-cuda/bench_grpo_loss_cuda --logits-only
```

The GPT-2 example is optional. CMake fetches one pinned `llm.c` revision; the
checkpoint is from its starter pack.

```bash
cmake -S . -B build-gpt2 \
  -DGRPO_USE_CUDA=ON -DGRPO_BUILD_GPT2=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpt2 -j
curl -L -o gpt2_124M.bin \
  https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_124M.bin
./build-gpt2/gpt2_grpo gpt2_124M.bin 4 uniform 42
```

## Reading the code

- [`src/cpu/grpo_loss.cpp`](src/cpu/grpo_loss.cpp) has the objective and its
  analytic derivative.
- [`examples/tiny_grpo_train.cpp`](examples/tiny_grpo_train.cpp) contains the
  complete toy policy and update loop.
- [`src/cuda/grpo_loss_cuda.cu`](src/cuda/grpo_loss_cuda.cu) has the selected
  log-probability kernels and the fused logits path.
- [`examples/gpt2_grpo.cu`](examples/gpt2_grpo.cu) connects that path to GPT-2.
- [`examples/qwen_train.cu`](examples/qwen_train.cu) handles Qwen generation,
  GRPO updates and checkpoints. The Countdown script supplies tokens and rewards.
- [`examples/gpt2_grpo_2gpu.cu`](examples/gpt2_grpo_2gpu.cu) adds synchronous
  NCCL gradient averaging.

For a selected token, the policy ratio is
`rho = exp(logp_new - logp_old)`. The policy term takes the smaller of
`rho * advantage` and its clipped version. The sampled KL penalty is
`exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1`.

The CPU implementation returns both the loss statistics and `dL/dlogp_new`.
The logits API applies the log-softmax Jacobian as well, producing the buffer
that a model backward pass expects. Sequence mean is the default reduction;
token and length-weighted reductions are also available.

## CUDA result

The model-facing baseline uses separate log-softmax/gather, GRPO, and logits
backward kernels. The fused kernel carries the row maximum and normalizer in
one pass, following the online-softmax recurrence.

These are median device-resident times on a Tesla T4, using 15 samples in
alternating order:

| rows | vocabulary | mask | separate | fused |
| ---: | ---: | --- | ---: | ---: |
| 1,024 | 32,768 | full | 2.3959 ms | 1.8687 ms |
| 1,024 | 32,768 | ragged | 1.8158 ms | 1.4404 ms |
| 1,024 | 131,072 | full | 10.3036 ms | 7.9044 ms |

The benchmark also contains a deliberately simple per-token atomic kernel and
a block-reduced version. Every CUDA path is checked against the CPU result
before it is timed.

## GPT-2 checks

The GPT-2 example uses eight short prompts, exact one-token rewards, a
fixed initial reference, and four policy passes per rollout batch. On an A100
PCIe 40 GB, one four-update run moved mean target probability from `0.034258`
to `0.338416`. A one-update run under Compute Sanitizer reported no errors.

For two GPUs, build with NCCL and run:

```bash
cmake -S . -B build-gpt2 \
  -DGRPO_USE_CUDA=ON -DGRPO_BUILD_GPT2=ON \
  -DGRPO_BUILD_DISTRIBUTED=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpt2 -j
./build-gpt2/gpt2_grpo_2gpu gpt2_124M.bin 4
```

Each process owns a model replica and eight rollouts per prompt. Rewards are
gathered before the advantages are computed; parameter gradients are averaged
after every policy pass. On two A100 PCIe 40 GB cards, both replicas moved from
`0.0597304` to `0.666268`, and their full parameter hashes matched afterward.

The GPT-2 examples sample on the CPU. The response-length and variance-guided
rollout experiments are in [`experiments/`](experiments/README.md).

## References

- [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- [`llm.c`](https://github.com/karpathy/llm.c)
- [`RL.cu`, the pinned Qwen model and kernels](https://github.com/KJLdefeated/RL.cu/tree/77ecd8385d9836bc8377e042f4e2b7dd0b6df9c6)
