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
- Six completed GPU checks total about $1.00 at listed rates. Image builds,
  an H100 startup failure and other startup charges are not in that estimate.
  The failed job was stopped and its remote-import bug fixed. L4 stayed queued.

The native check now passes (`native-parity-02`, `e813a7d`): all 310 parameter
tensors, padding, microbatches, FP32 weight loading and zero gradients. Relative
gradient error is 0.56% with loss scaling, versus 1.55% without. CUDA loss checks,
the repeated-token regression and memcheck pass. This covers one two-sequence
fixture, not optimizer updates or native learning.

The first run failed the repeat check because of FP16 embedding atomics. Summing
repeated rows in FP32 cut that difference from 0.005% to 0.0000022%, with the
same fixture and tolerances. Both runs and the plot are in the results bundle.
The reduction suits short microbatches; large batches will need a faster one.

Clipped Adam and model-only checkpoint export are implemented, with runtime
checks still pending. Next: compare three updates with PyTorch and verify native
and Transformers reloads. Gradients stay scaled until Adam reads them as FP32;
the norm excludes attention scratch. Loading preserves original master weights.

Native learning, the 24 GB run, final testing and the repeat seed remain open.
This chat owns integration. Budget: $5 pilot, $15 native comparison, $5 repeat,
$5 contingency. Cost figures above are estimates, not billing receipts.
