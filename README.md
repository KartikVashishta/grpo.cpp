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
