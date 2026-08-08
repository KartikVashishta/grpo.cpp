# grpo.cpp

I started this repo to make the GRPO loss small enough to inspect without a
framework around it.

The CPU path computes grouped advantages, PPO clipping, the sampled reverse-KL
term, and an analytic derivative with respect to each sampled log probability.
The tests cover the clipping branches, variable response lengths, finite
differences, and bad inputs.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/grpo_loss_cpu
```
