# Qwen3-0.6B on Countdown

This is `native-a10-01`, seed 42, after 64 GRPO updates in grpo.cpp.
The base model is [Qwen3-0.6B at c1899de](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca).
All model weights were fine-tuned on three-number Countdown. The tokenizer is unchanged.

Held-out success went from 26/512 to 198/512 (5.1% to 38.7%), with one sampled
answer per puzzle. The A10 training function took 143 seconds, about $0.053
of estimated compute. Builds, startup, storage, network and the separate final
test are excluded. This result measures the exact expression task, not general
mathematical reasoning.

[Recorded answers and method](https://github.com/KartikVashishta/grpo.cpp/tree/main/experiments/countdown).
Training source: `27097098e2c7ebde83a979ab8032949a6f9f7f92`.
Final-test source: `141b3de`.

The archive contains the original FP32 weights, model configuration and tokenizer.
It has no optimizer state. The model uses the included Apache-2.0 license.

SHA-256 of `model.safetensors`:

```text
bf8e8c5984a95705859a4e4e19c510369b8d54e538292bf29327fe3ef399aa9c
```

Use the repository's `experiments/countdown/sample.py` to apply the same prompt
and sampling settings. It accepts a local checkpoint directory, `--numbers`
and `--target`. Answers are sampled; the example shown in the demo is a saved
result, not a guarantee for a new sample.
