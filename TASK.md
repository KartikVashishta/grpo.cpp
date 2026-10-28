# Countdown

The one-GPU Qwen3-0.6B experiment is finished. A10 and H100 each have two
training seeds, saved checkpoints and held-out tests. Native was faster;
test scores varied. [Results and commands](experiments/countdown/README.md).

The 25 GPU runs used about $2.54 of estimated compute against the $30 budget,
excluding builds, startup, storage and network. All jobs stopped on 28 October
2026. Raw answers and measurements are in `experiments/countdown/results.tar.gz`.

Trimmed the docs and redrew the plots. The desktop and mobile viewers work;
curve values, timing segments, table values and saved answers are unchanged.
