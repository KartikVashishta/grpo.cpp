# Countdown

Get Qwen3-0.6B learning three-number Countdown, then build and compare the
native training path. A finished run needs a reloadable checkpoint, held-out
scores, timing, memory and cost. Repeat it with another seed before claiming
an advantage. Model and data revisions are in `experiments/countdown/config.json`.

28 October:

- Started from `0f78dd1`. Existing C++ tests pass.
- Verifier, fixed splits and reference runner are in place. Four Python tests
  pass. No model samples or learning result yet.
- Fixed a PyArrow dependency conflict before GPU allocation.
- L4 job `ap-U7z9D7nQuZLJb43OEzmns1` stayed queued and was stopped.
- H100 job `ap-r6cZNRRJS2zk2t2plMDczV` failed during module import. Stopped it
  and fixed the remote source path; the import check now passes locally.
- RL.cu is a possible backend, but its checkpoint saver is a stub and its
  default optimizer/KV allocation order needs fixing for 24 GB.

Next: retry calibration, then check one training update and checkpoint reload.
If those work, run the short training pilot. Native integration, final test
evaluation and the 24 GB target remain open. This chat owns integration.

Compute budget: $5 pilot, $15 native comparison, $5 repeat, $5 contingency.
Record failed startup costs too; rate estimates are not billing receipts.
