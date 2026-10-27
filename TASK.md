# Countdown

Get Qwen3-0.6B learning three-number Countdown, then build and compare the
native training path. A finished run needs a reloadable checkpoint, held-out
scores, timing, memory and cost. Repeat it with another seed before claiming
an advantage. Model and data revisions are in `experiments/countdown/config.json`.

28 October:

- Started from `0f78dd1`. Existing C++ tests pass.
- Verifier, fixed splits and reference runner are in place. Five Python tests
  pass, including remote module import.
- Fixed a PyArrow dependency conflict before GPU allocation.
- L4 job `ap-U7z9D7nQuZLJb43OEzmns1` stayed queued and was stopped.
- H100 job `ap-r6cZNRRJS2zk2t2plMDczV` failed during module import. Stopped it
  and fixed the remote source path; the import check now passes locally.
- `calibration-h100-02`: 0/1,024 correct; 1,017 parse failures and seven wrong
  number sets. No mixed-reward groups. 86 seconds, about $0.10 at listed rates.
  Raw answers and plots are under `runs/countdown/`. Training has not started.
- `calibration-h100-03`: adding two format examples gave 36/1,024 correct and
  29/128 mixed-reward groups. Mean length fell from 36.3 to 16.8 tokens.
  69 seconds, about $0.08. Keep this prompt for the training pilot.
- RL.cu is a possible backend, but its checkpoint saver is a stub and its
  default optimizer/KV allocation order needs fixing for 24 GB.

Next: check one training update and checkpoint reload.
If those work, run the short training pilot. Native integration, final test
evaluation and the 24 GB target remain open. This chat owns integration.

Compute budget: $5 pilot, $15 native comparison, $5 repeat, $5 contingency.
Record failed startup costs too; rate estimates are not billing receipts.
