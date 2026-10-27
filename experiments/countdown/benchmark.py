"""Time one recorded token batch with native code and TRL."""

import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tarfile
import time

from native import Native, prompt_ids
from parity import prepare
from task import messages, verify


def torch_batch(model_dir, fixture):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from trl.kernels.chunked_logprob import _use_triton
    from trl.trainer.utils import FusedCausalLMOutput

    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32,
                                                attn_implementation="sdpa").cuda()
    args = GRPOConfig(output_dir="/tmp/grpo-fixed-work", per_device_train_batch_size=2,
        gradient_accumulation_steps=16, num_generations=8, max_steps=1,
        temperature=1., top_p=1., top_k=0, max_completion_length=64,
        beta=0., loss_type="grpo", scale_rewards="none", num_iterations=1,
        bf16=True, gradient_checkpointing=True, use_vllm=False,
        learning_rate=1e-6, lr_scheduler_type="constant", optim="adamw_torch",
        report_to="none", save_strategy="no", logging_strategy="no", disable_tqdm=True)
    trainer = GRPOTrainer(model=model, args=args, processing_class=tokenizer,
        reward_funcs=lambda completions, **kwargs: [0.] * len(completions),
        train_dataset=Dataset.from_list([{"prompt": messages(p)} for p in fixture["puzzles"]]))
    trainer.current_gradient_accumulation_steps = 16
    assert trainer.temperature == 1.
    model = trainer.model
    model.gradient_checkpointing_enable()
    model.train()
    assert model.is_gradient_checkpointing and all(p.dtype == torch.float32 for p in model.parameters())
    assert _use_triton(torch.device("cuda"))
    # Retain Adam state as in a live trainer, without changing any weight.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0., foreach=False, fused=False)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    length = max(len(p)+len(c) for p, c in zip(fixture["prompts"], fixture["completions"], strict=True))
    rewards = torch.tensor(fixture["rewards"], device="cuda").reshape(4, 8)
    advantages = (rewards-rewards.mean(1, keepdim=True)).flatten()
    batches = []
    for start in range(0, 32, 2):
        prompts = fixture["prompts"][start:start+2]
        assert prompts[0] == prompts[1]
        width = length-len(prompts[0])
        ids = torch.zeros((2, width), device="cuda", dtype=torch.long)
        mask = torch.zeros_like(ids)
        for b, completion in enumerate(fixture["completions"][start:start+2]):
            ids[b, :len(completion)] = torch.tensor(completion, device="cuda")
            mask[b, :len(completion)] = 1
        batches.append({"prompt_ids": torch.tensor(prompts, device="cuda"),
                        "prompt_mask": torch.ones((2, len(prompts[0])), device="cuda", dtype=torch.long),
                        "completion_ids": ids, "completion_mask": mask,
                        "advantages": advantages[start:start+2]})

    calls = 0
    def confirm_fused(module, args, kwargs, output):
        nonlocal calls
        assert kwargs.get("fused_lm_head") is True and isinstance(output, FusedCausalLMOutput)
        assert output.log_probs is not None and output.entropy is not None
        calls += 1

    hook = model.register_forward_hook(confirm_fused, with_kwargs=True)
    samples = []
    for iteration in range(7):
        torch.cuda.synchronize()
        started = time.perf_counter()
        model.zero_grad(set_to_none=True)
        for batch in batches:
            # compute_loss already divides each microbatch by 16.
            loss = trainer.compute_loss(model, batch)
            loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        if iteration == 0:
            assert calls == 16
            hook.remove()
        if iteration == 1:
            torch.cuda.reset_peak_memory_stats()
        if iteration >= 2:
            samples.append(elapsed)
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    free, total = torch.cuda.mem_get_info()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True).item()
    result = {"seconds": samples, "gradient_norm": norm, "fused_head_verified": calls == 16,
              "allocated_bytes": allocated, "reserved_bytes": reserved,
              "sampled_device_used_bytes": total-free, "backbone_sequence": length}
    del loss, model, trainer, optimizer, batches, batch, parameter, ids, mask, rewards, advantages
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(output, provenance):
    from transformers import AutoTokenizer

    started = time.perf_counter()
    model_dir, _, _, _ = prepare(output, provenance, limits={"gradient_norm_relative_gap": .10},
        reference_description="Fixed recorded token batch; native and actual TRL compute_loss/backward, weights unchanged")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    with tarfile.open("/experiment/results.tar.gz") as archive:
        raw = archive.extractfile("native-h100-01/rollouts.jsonl").read()
        records = [json.loads(row) for row in raw.splitlines()]
        records = [row for row in records if row["step"] == 1]
        training = [json.loads(row) for row in archive.extractfile("native-h100-01/training.jsonl").read().splitlines()]
    assert len(records) == 32 and training[0]["gradient_norm"] == 0
    assert all(row["reward"] == verify(row["text"], row["nums"], row["target"]).reward for row in records)
    fixture = {"prompts": prompt_ids(tokenizer, records), "completions": [r["token_ids"] for r in records],
               "rewards": [r["reward"] for r in records], "puzzles": records[::8]}
    assert all(len(set(tuple(p) for p in fixture["prompts"][i:i+8])) == 1 for i in range(0,32,8))
    (output / "batch.json").write_text(json.dumps(fixture, indent=2) + "\n")
    metadata = json.loads((output / "metadata.json").read_text())
    metadata.update(binary_sha256=hashlib.sha256(Path("/build/qwen_train").read_bytes()).hexdigest(),
        rollout_source="native-h100-01, step 1 before its first nonzero update",
        rollout_sha256=hashlib.sha256(raw).hexdigest(), warmup=2, samples_per_round=5, rounds=3,
        timed_work="forward, GRPO loss and backward; no generation, clipping or optimizer step",
        differences=["Native FP16 versus TRL BF16 autocast with FP32 parameters",
                     "TRL includes entropy and clipping metrics",
                     "TRL computes one extra final-token backbone position per row",
                     "Native includes small host copies and stage synchronizations"],
        tokens=sum(len(row) for row in fixture["completions"]),
        microbatch=2, gradient_accumulation=16, group_size=8)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    subprocess.run(["/build/test_qwen_optimizer"], check=True, timeout=60,
                   stdout=(output / "optimizer-check.txt").open("w"), stderr=subprocess.STDOUT)
    results = []
    for round in range(3):
        for implementation in (("native", "trl") if round % 2 == 0 else ("trl", "native")):
            if implementation == "native":
                with Native(model_dir, output / f"native-{round}.log") as engine:
                    result = engine.call("benchmark", prompts=fixture["prompts"], completions=fixture["completions"],
                                         rewards=fixture["rewards"], group_size=8, warmup=2, iterations=5)
                assert result["optimizer_step"] == 0
                assert abs(result["gradient_norm"] / training[1]["gradient_norm"] - 1) < 1e-4
                result["seconds"] = [r["seconds"] for r in result["samples"]]
            else:
                result = torch_batch(model_dir, fixture)
            result.update(round=round, implementation=implementation)
            results.append(result)
            (output / "samples.json").write_text(json.dumps(results, indent=2) + "\n")
            print(implementation, round, statistics.median(result["seconds"]), flush=True)
    summary = {"passed": True, "seconds": time.perf_counter()-started}
    for name in ("native", "trl"):
        seconds = [s for row in results if row["implementation"] == name for s in row["seconds"]]
        norms = [row["gradient_norm"] for row in results if row["implementation"] == name]
        summary[name] = {"median_seconds": statistics.median(seconds), "min_seconds": min(seconds),
                         "max_seconds": max(seconds), "samples": len(seconds), "gradient_norms": norms}
    gaps = [abs(a/b-1) for a, b in zip(summary["native"]["gradient_norms"],
                                      summary["trl"]["gradient_norms"], strict=True)]
    summary["gradient_norm_relative_gaps"] = gaps
    summary["passed"] = max(gaps) <= .10
    summary["time_ratio"] = summary["trl"]["median_seconds"] / summary["native"]["median_seconds"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
