# Countdown

Build the one-GPU Qwen3-0.6B training path and compare it with TRL on fixed
Countdown puzzles. Finish with usable checkpoints, held-out scores, timing,
memory, cost and a repeat seed. Pins are in `experiments/countdown/config.json`.

28 October:

- The two-example prompt gave 36/1,024 correct calibration responses and
  29/128 mixed-reward groups. The original prompt had no correct responses.
- TRL (`reference-h100-01`, `d2b2ee7`) reached 107/256 development answers
  after 64 updates, up from 10/256. Function time: 397.5 seconds, about $0.46.
- Native (`native-h100-01`, `12d5c57`) reached 87/256 after 64 updates, up from
  15/256: 80 gains and eight regressions. Function time: 77.2 seconds, about
  $0.09. Reload reproduced all sampled tokens on 32 development puzzles.
- Native gradients (`native-parity-02`, `e813a7d`) differ from FP32 PyTorch
  by 0.56% on the fixed fixture. Three Adam updates (`native-update-01`,
  `1116756`) match within 1.2e-7. Saved weights reload in native and HF models.
- Generation (`native-generation-04`, `a3b8a18`) passes cached/full-prefix
  scoring, EOS, an update with controlled rewards, reload and memcheck. Maximum
  log-probability difference from PyTorch: 0.0082. This is not a learning run.

Two kernel fixes mattered: sum repeated embedding gradients before rounding,
and keep sampler uniforms below 1. Greedy output can change with batch size
at FP16 ties; the check verifies actual argmax choices and probability error.
Earlier failures remain in `experiments/countdown/results.tar.gz`, along with
the measurements. The README has the plots and commands.

The native run is faster with a lower score. Training token counts were close
(26,305 native, 26,709 TRL), but RNG and compute precision differ. Next: a
fixed-work comparison, 24 GB run and repeat seed, then final testing. This chat
owns integration.

Thirteen completed GPU runs total about $1.36 at listed rates, excluding image
builds and startup. No GPU job is left running. Budget: $5 pilot, $15 native
comparison, $5 repeat, $5 contingency. These are estimates, not billing receipts.
