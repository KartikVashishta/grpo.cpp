# Countdown

Build a small one-GPU Qwen3-0.6B trainer with usable checkpoints, held-out
scores, timing, memory, cost and repeat seeds. The recipe is fixed in
`experiments/countdown/config.json`.

28 October — finished:

- Two native A10 runs reached 198/512 and 179/512 on the frozen final test,
  up from 26 and 14. Training took 143 and 131 seconds, about $0.05 each.
  TRL reached 183/512 and 199/512 in 439 and 392 seconds. Native was faster;
  the test-score advantage changed sides (`test-a10-01`, source `141b3de`).
- Both implementations fit a 24 GB GPU. Native sampled device usage was
  10.27 GiB; TRL reserved 11.99 GiB. The memory measures differ.
- Both H100 seeds also learned. Native reached 174/512 and 196/512;
  TRL reached 214/512 and 202/512 (`test-h100-01`, source `db6b1b5`).
- The same recorded batch took 0.440 seconds through native forward/loss/
  backward and 1.965 seconds through TRL (`fixed-work-h100-02`, `9b97b95`).
  Precision and bookkeeping differ; this is not a quality comparison.
- Gradient and Adam checks passed against PyTorch. Generation, reload and
  memcheck passed on H100, L4 and A10. All eight trained checkpoints reproduced
  the saved tokens on reload. No recipe or checkpoint was chosen from test scores.

Raw answers and measurements, including failed attempts, are in
`experiments/countdown/results.tar.gz`. The README has the plots and commands
to rebuild the answer viewer. The downloaded model at
`runs/countdown/native-a10-01/final/` matches the weights used in the final test.

Twenty-five GPU functions total about $2.54 at recorded rates, excluding builds,
startup, storage and network, against a $30 budget. All GPU jobs have stopped.

The checkpoints save model weights, not optimizer state. Two seeds do not prove
equal learning quality or a general advantage over other trainers.
