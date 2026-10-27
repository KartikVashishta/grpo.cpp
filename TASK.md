# Qwen Countdown

Build a small one-GPU RL engine around a result someone can reproduce: teach
Qwen3-0.6B to solve three-number Countdown, then measure held-out accuracy,
time, memory and cost. Keep the C++ loss examples useful and the training path
easy to follow.

## What counts as done

- A runnable reference experiment with a strict verifier, fixed data splits,
  pinned dependencies and ordinary saved checkpoints.
- A native training path with checked forward values and gradients, generated
  prefix scoring, EOS masks, checkpoint export and held-out evaluation.
- Real learning curves, stage timings, peak memory and recorded answers. Show
  matched-work performance separately from changes in answer length.
- A repeat run and final test evaluation before claiming a practical advantage.
  A script delivered is not a successful run; a successful run is not yet a
  competitive result.

## First gate

Use the non-thinking Qwen3-0.6B checkpoint, three input numbers, one expression,
64 generated tokens and eight completions per prompt. Only an exact solution
using every input once earns reward. Keep permutations of the same numbers and
target together across splits.

First sample 128 calibration prompts. Record correctness, parse failures,
length, truncation and groups with both successes and failures. Then run a
short established-trainer pilot and measure development accuracy and runtime.
Do not tune on the final test set. Change the calibration recipe if the model
has too little reward variation to learn.

Keep the proposed compute caps: $5 for calibration and the reference pilot,
$15 for native comparison, $5 for an independent rerun and $5 contingency.
Record rate-based cost estimates separately from provider billing. Bound each
job before starting it and keep credentials outside this repository.

## Current evidence — 28 October 2026

- Starting source: `0f78dd1` (`trim the first release`); working tree was clean.
- Existing implementation: CPU/CUDA GRPO loss, a toy policy, optional one-token
  GPT-2 examples and two small research experiments. No Qwen or Countdown path.
- Model pin: `Qwen/Qwen3-0.6B` at
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- Data pin: `Jiayi-Pan/Countdown-Tasks-3to4` at
  `408f70d177020686d34a56bba5952feb45aaaee4`.
- Delivered: strict exact-arithmetic verifier, frozen split manifest, pinned
  reference runner and bounded Modal entrypoint. Verifier and split tests pass;
  all three existing C++ tests pass in a fresh Release build with experiments.
- Reviewed: independent loss/backend/API audits and a pre-dispatch runtime
  review. Corrected Modal resource limits and protected existing run records.
  No new GPU behavior demonstrated yet.
- Open: calibration, reference learning, native model integration and all
  performance claims.
- Next: run the bounded calibration, and use
  the observed mixed-reward groups to decide the pilot configuration.
- First image build stopped before GPU allocation: Datasets 5.1 requires
  PyArrow 24 or newer. Updated the explicit PyArrow pin to 24.0.0; this was a
  dependency resolution failure, not a model or learning result.
- The corrected image resolved successfully. Calibration app
  `ap-U7z9D7nQuZLJb43OEzmns1` remained queued with no running L4 container.
  Stopped it before trying the authorized H100 with a 20-minute timeout.
  An H100 result will not establish the planned 24 GB memory target.

The main thread owns integration. Independent read-only reviews cover existing
loss correctness, the candidate native backend and reference-trainer APIs.
