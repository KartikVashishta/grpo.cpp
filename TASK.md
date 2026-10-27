# Countdown

Build a small one-GPU Qwen3-0.6B trainer, with usable checkpoints, held-out
scores, timing, memory, cost and repeat seeds. Keep the fixed Countdown recipe
in `experiments/countdown/config.json`.

28 October:

- Both H100 seeds finished 64 updates. Native took 77 and 89 seconds;
  TRL took 398 and 242 seconds. All four checkpoint reload checks passed.
- The frozen final test gave native 174/512 and 196/512, up from 25 and 12.
  TRL reached 214/512 and 202/512, up from 20 and 15. No checkpoint was
  selected or recipe changed using these test results (`test-h100-01`, `db6b1b5`).
- Replaying identical token IDs and rewards took a median 0.440 seconds
  native and 1.965 seconds TRL for forward/loss/backward (`fixed-work-h100-02`,
  `9b97b95`). Fifteen samples each; weights fixed. Precision and bookkeeping
  differ. The 4.92% norm gap checks scaling, not full-gradient parity.
- L4 and A10 passed generation, backward, reload and memcheck. The A10
  reference training run is finishing; the matched native run comes next.

The native runs learn and cost less here, but TRL has higher final-test scores.
Raw answers, earlier failures and measurements stay in `results.tar.gz`.
The report shows both seeds and lets you inspect final-test failures too.

Next: finish the A10 comparison and its final test, update the plots and check
the run instructions. H100 results do not establish 24 GB training performance.

Twenty completed GPU runs total about $2.06 at listed rates, excluding builds
and startup. Budget caps remain $5 pilot, $15 native comparison, $5 repeat and
$5 contingency.
