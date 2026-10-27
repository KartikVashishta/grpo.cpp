"""Check cached generation, full-prefix scoring, EOS and a complete update."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

from native import Native, eos_ids, prompt_ids
from parity import prepare
from task import verify


LIMITS = {"cached_logp_max_abs": .05, "cached_logp_rms": .015,
          "torch_logp_max_abs": .05, "torch_logp_rms": .015}


def run(output, provenance):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    model_dir, _, _, _ = prepare(output, provenance, limits=LIMITS,
        reference_description="Native cached generation against full-prefix native and rounded-weight FP32 Transformers scoring")
    metadata = json.loads((output / "metadata.json").read_text())
    metadata["binary_sha256"] = hashlib.sha256(Path("/build/qwen_train").read_bytes()).hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    subprocess.run(["/build/test_qwen_sampler"], check=True, timeout=60,
                   stdout=(output / "sampler.txt").open("w"), stderr=subprocess.STDOUT)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    puzzles = [{"nums": [43, 44, 58], "target": 57}, {"nums": [17, 57, 81], "target": 41},
               {"nums": [2, 3, 10], "target": 35}, {"nums": [1, 11, 99], "target": 9}]
    prompts = prompt_ids(tokenizer, puzzles)
    # Include substantially different lengths, and an EOS before the token cap.
    varied = [tokenizer.encode("Hello", add_special_tokens=False), *prompts[:3]]
    settings = {"max_new_tokens": 20, "temperature": 0., "seed": 42, "eos_ids": eos_ids(model_dir)}
    summary = {"passed": False, "limits": LIMITS}
    with Native(model_dir, output / "native.log") as engine:
        generated = engine.call("generate", prompts=varied, trace=True, **settings)
        scored = engine.call("score", prompts=varied, completions=generated["completions"])
        # Force the first sequence to finish on its first token, so decoding compacts slots.
        stop = generated["completions"][0][0]
        compact = engine.call("generate", prompts=varied, trace=True,
                              **{**settings, "eos_ids": settings["eos_ids"] + [stop]})
        assert len(compact["completions"][0]) == 1 and compact["active_sizes"][1] < len(varied)
        compact_score = engine.call("score", prompts=varied, completions=compact["completions"])
        records = {"prompts": varied, "generated": generated, "scored": scored,
                   "compact": compact, "compact_score": compact_score}
        (output / "generation.json").write_text(json.dumps(records, indent=2) + "\n")
        diffs = []
        for generation, score in [(generated, scored), (compact, compact_score)]:
            diffs.extend(a-b for left, right in zip(generation["logps"], score["logps"], strict=True)
                         for a, b in zip(left, right, strict=True))
        error = np.asarray(diffs)
        summary["cached_logp_max_abs"] = float(abs(error).max())
        summary["cached_logp_rms"] = float(np.sqrt(np.mean(error**2)))
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        assert summary["cached_logp_max_abs"] <= LIMITS["cached_logp_max_abs"]
        assert summary["cached_logp_rms"] <= LIMITS["cached_logp_rms"]
        summary["compaction_checked"] = True
        # Greedy batches and single-sequence decoding must follow the same prefix.
        singles = [engine.call("generate", prompts=[row], **settings)["completions"][0] for row in varied]
        summary["greedy_batch_matches_singles"] = singles == generated["completions"]
        assert summary["greedy_batch_matches_singles"]
        training_prompts = [row for row in prompts for _ in range(8)]
        rollout = engine.call("generate", prompts=training_prompts,
                              **{**settings, "max_new_tokens": 64, "temperature": 1.})
        repeated = engine.call("generate", prompts=training_prompts,
                               **{**settings, "max_new_tokens": 64, "temperature": 1.})
        assert repeated["completions"] == rollout["completions"]
        rewards = [verify(tokenizer.decode(ids, skip_special_tokens=True), puzzles[i//8]["nums"],
                          puzzles[i//8]["target"]).reward for i, ids in enumerate(rollout["completions"])]
        # Fixed mixed rewards also exercise an update if this small sample earns no real reward.
        controlled = [1., 0., 0., 0., 0., 0., 0., 0.] * 4
        update = engine.call("update", rewards=controlled, group_size=8)
        assert update["step"] == 1 and np.isfinite(update["gradient_norm"]) and update["gradient_norm"] > 0
        assert update["advantages"] == [.875, -.125, -.125, -.125, -.125, -.125, -.125, -.125] * 4
        after = engine.call("score", prompts=training_prompts, completions=rollout["completions"])
        records.update(rollout=rollout, update=update, after=after)
        (output / "generation.json").write_text(json.dumps(records, indent=2) + "\n")
        gain = sum(a * (np.mean(new)-np.mean(old)) for a, new, old in
                   zip(update["advantages"], after["logps"], update["logps"], strict=True)) / 32
        assert gain > 0
        summary.update(gradient_norm=update["gradient_norm"], weighted_logp_gain=float(gain),
                       real_rewards=rewards, update_rewards=controlled,
                       peak_device_used_bytes=engine.peak_used)
        engine.call("save", path=str(output / "final"))
        before_reload = engine.call("generate", prompts=prompts, **settings)["completions"]
    with Native(output / "final", output / "reload.log") as engine:
        after_reload = engine.call("generate", prompts=prompts, **settings)["completions"]
    assert before_reload == after_reload
    summary["reload_tokens_match"] = True
    with Native(model_dir, output / "memcheck.log", memcheck=True) as engine:
        engine.call("generate", prompts=[prompts[0]] * 2,
                    **{**settings, "max_new_tokens": 4})
        zero = engine.call("update", rewards=[0., 0.], group_size=2)
        assert zero["gradient_norm"] == 0
    assert "ERROR SUMMARY: 0 errors" in (output / "memcheck.log").read_text()
    summary["memcheck_passed"] = True
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Check the original generated prefixes independently of the cache implementation.
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32,
                                                attn_implementation="eager").to("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 2:
                parameter.copy_(parameter.half().float())
        diffs = []
        for prompt, completion, native in zip(varied, generated["completions"], scored["logps"], strict=True):
            ids = torch.tensor([prompt + completion], device="cuda")
            logits = model(ids[:, :-1], use_cache=False).logits.float()
            values = logits.log_softmax(-1).gather(-1, ids[:, 1:, None])[0, len(prompt)-1:, 0].cpu().tolist()
            diffs.extend(a-b for a, b in zip(native, values, strict=True))
    error = np.asarray(diffs)
    summary["torch_logp_max_abs"] = float(abs(error).max())
    summary["torch_logp_rms"] = float(np.sqrt(np.mean(error**2)))
    summary["passed"] = (summary["torch_logp_max_abs"] <= LIMITS["torch_logp_max_abs"]
                         and summary["torch_logp_rms"] <= LIMITS["torch_logp_rms"])
    summary["seconds"] = time.perf_counter() - started
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
