# Countdown

The one-GPU Qwen3-0.6B experiment is finished. A10 and H100 each have two
training seeds, saved checkpoints and held-out tests. Native was faster;
test scores varied. [Results and commands](experiments/countdown/README.md).

The 26 GPU runs used about $2.55 of estimated compute against the $30 budget,
excluding builds, startup, storage and network. All jobs stopped on 28 October
2026. Raw answers and measurements are in `experiments/countdown/results.tar.gz`.

The [viewer](https://kartikvashishta.github.io/grpo.cpp/) and
[checkpoint with a 28-second replay](https://github.com/KartikVashishta/grpo.cpp/releases/tag/countdown-a10)
are public. GitHub's file hashes match the local release assets; anonymous
downloads work. The README, About section and social preview are updated.
All 8,192 final-test verdicts rechecked. A clean clone and empty Modal environment
passed the build, one A10 update and checkpoint reload; receipts are in the
archive under `first-run-01`. All eight local tests and the live viewer checks
passed. The temporary Modal environment was deleted.
