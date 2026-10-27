# Countdown

Build the one-GPU Qwen3-0.6B training path and compare it with TRL on fixed
Countdown puzzles. Finish with usable checkpoints, held-out scores, timing,
memory, cost and a repeat seed. Pins are in `experiments/countdown/config.json`.

28 October:

- The original prompt scored 0/1,024. Two format examples gave 36/1,024 and
  29/128 mixed-reward groups. The verifier and data splits stayed fixed.
- `reference-h100-01` (`d2b2ee7`) finished 64 updates: development accuracy
  rose from 10/256 to 107/256. Correctness among valid expressions rose from
  16.1% to 46.5%; this compares different subsets of answers.
- The function ran for 397.5 seconds, reserving 12.83 GiB. Estimated compute:
  $0.46. Final weights reloaded with identical tokens on 32 development puzzles.
- Raw answers and measurements are in `experiments/countdown/results.tar.gz`;
  the README has the plots. Five Python tests and three C++ tests pass.
- Four completed GPU checks total about $0.71 at listed rates. Image builds,
  an H100 startup failure and other startup charges are not in that estimate.
  The failed job was stopped and its remote-import bug fixed. L4 stayed queued.

The native comparison is written; CUDA compilation and numerical results are
still pending. It checks forward values, all gradients, loss scaling, padding,
microbatches and FP32 weight loading against PyTorch. RL.cu is pinned at
`77ecd8385d9836bc8377e042f4e2b7dd0b6df9c6`, with loader validation patched.
Next: finish the CPU build, then run the bounded H100 comparison. Native
checkpoint export still needs implementing; the upstream saver is a stub.

Native learning, the 24 GB run, final testing and the repeat seed remain open.
This chat owns integration. Budget: $5 pilot, $15 native comparison, $5 repeat,
$5 contingency. Cost figures above are estimates, not billing receipts.
