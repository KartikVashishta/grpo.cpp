"""A measured Transformers/TRL reference run. This is not the native engine."""

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from task import json_bytes, messages, verify


def append_json(path, value):
    with path.open("a") as file:
        file.write(json.dumps(value) + "\n")


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["id"]].append(row["reward"])
    count = len(records)
    return {
        "completions": count, "prompts": len(groups),
        "accuracy": sum(r["reward"] for r in records) / count,
        "mixed_groups": sum(0 < sum(g) < len(g) for g in groups.values()),
        "mean_tokens": sum(r["tokens"] for r in records) / count,
        "generated_tokens": sum(r["tokens"] for r in records),
        "truncated": sum(not r["eos"] for r in records),
        "statuses": dict(Counter(r["status"] for r in records)),
    }


class Timings:
    """Exclusive synchronized wall times; nested stages are not counted twice."""

    def __init__(self):
        self.seconds = defaultdict(float)
        self.stack = []

    @contextmanager
    def measure(self, name):
        import torch
        torch.cuda.synchronize()
        frame = [time.perf_counter(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - frame[0]
            self.stack.pop()
            self.seconds[name] += elapsed - frame[1]
            if self.stack:
                self.stack[-1][1] += elapsed


def sample(model, tokenizer, puzzles, config, step, split, count=1):
    import torch

    records = []
    was_training = model.training
    model.eval()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    # Evaluation must not advance training's random-number stream.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(config["seed"] + 10000)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(puzzles), 32 // count):
                batch = puzzles[start:start + 32 // count]
                prompts = [tokenizer.apply_chat_template(messages(p), tokenize=False,
                           add_generation_prompt=True, enable_thinking=False) for p in batch]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                                   add_special_tokens=False).to(model.device)
                outputs = model.generate(
                    **inputs, do_sample=True, num_return_sequences=count,
                    max_new_tokens=config["max_new_tokens"],
                    temperature=config["temperature"], top_p=config["top_p"],
                    top_k=config["top_k"], repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id, use_cache=True,
                )[:, inputs["input_ids"].shape[1]:].tolist()
                for index, ids in enumerate(outputs):
                    puzzle = batch[index // count]
                    end = next((i + 1 for i, token in enumerate(ids) if token in eos_ids), len(ids))
                    ids = ids[:end]
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    verdict = verify(text, puzzle["nums"], puzzle["target"])
                    records.append({**puzzle, "step": step, "split": split,
                                    "sample": index % count, "text": text,
                                    "token_ids": ids, "tokens": len(ids),
                                    "eos": bool(ids and ids[-1] in eos_ids),
                                    **verdict.as_dict(), "reward": verdict.reward})
    model.train(was_training)
    return records


def run(config, data_dir, output, mode, provenance, max_seconds=3000):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    frozen = json.loads(Path(__file__).with_name("data_manifest.json").read_text())
    if manifest != frozen:
        raise ValueError("Data manifest does not match the frozen experiment")
    splits = {}
    for name in ("calibration", "dev", "train"):
        content = (data_dir / f"{name}.jsonl").read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["splits"][name]["sha256"]:
            raise ValueError(f"Split hash mismatch: {name}")
        splits[name] = [json.loads(line) for line in content.splitlines()]
    metadata = {
        "config": config, "mode": mode, "source": provenance, "data": manifest,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "trl", "accelerate", "datasets")},
        "installed_packages": sorted(f"{p.metadata['Name']}=={p.version}"
                                     for p in importlib.metadata.distributions()),
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "timing_method": "synchronized wall time; nested stages excluded from parents",
        "cost_method": "resource-rate estimate, not a provider invoice",
        "rates_usd_per_second": {"gpu": 0.000222, "cpu_core": 0.0000131, "ram_gib": 0.00000222},
        "cpu_cores": 2, "ram_gib": 16, "max_seconds": max_seconds,
    }
    (output / "metadata.json").write_bytes(json_bytes(metadata))
    set_seed(config["seed"])
    timings = Timings()
    with timings.measure("load"):
        tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["model_revision"],
                                                 padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config["model"], revision=config["model_revision"], dtype=torch.float32,
            attn_implementation="sdpa").to("cuda")
    torch.cuda.reset_peak_memory_stats()

    def evaluate(step, split, count=1):
        with timings.measure("evaluation"):
            records = sample(model, tokenizer, splits[split], config, step, split, count)
        for record in records:
            append_json(output / "answers.jsonl", record)
        metrics = {"step": step, "split": split, **summarize(records),
                   "elapsed_seconds": time.perf_counter() - started,
                   "stage_seconds": dict(timings.seconds),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        append_json(output / "evaluations.jsonl", metrics)
        print(json.dumps(metrics), flush=True)
        return records

    if mode == "calibrate":
        evaluate(0, "calibration", config["group_size"])
    elif mode == "train":
        evaluate(0, "dev")

        def reward(completions, nums, target, id, completion_ids, trainer_state, **kwargs):
            rewards = []
            for index, (completion, numbers, goal, key, ids) in enumerate(
                    zip(completions, nums, target, id, completion_ids, strict=True)):
                text = completion[0]["content"]
                verdict = verify(text, numbers, goal)
                rewards.append(verdict.reward)
                append_json(output / "rollouts.jsonl", {
                    "step": trainer_state.global_step, "id": key, "nums": numbers,
                    "target": goal, "sample": index % config["group_size"],
                    "text": text, "tokens": len(ids), **verdict.as_dict(), "reward": verdict.reward})
            return rewards

        class MeasuredTrainer(GRPOTrainer):
            def _generate(self, *args, **kwargs):
                with timings.measure("generation"):
                    return super()._generate(*args, **kwargs)

            def _calculate_rewards(self, *args, **kwargs):
                with timings.measure("reward"):
                    return super()._calculate_rewards(*args, **kwargs)

            def _get_per_token_logps_and_entropies(self, *args, **kwargs):
                with timings.measure("scoring_forward"):
                    return super()._get_per_token_logps_and_entropies(*args, **kwargs)

            def compute_loss(self, *args, **kwargs):
                with timings.measure("loss"):
                    return super().compute_loss(*args, **kwargs)

            def training_step(self, *args, **kwargs):
                with timings.measure("backward_and_batch_overhead"):
                    return super().training_step(*args, **kwargs)

        class Measurements(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                self.optimizer_started = time.perf_counter()

            def on_optimizer_step(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                timings.seconds["optimizer"] += time.perf_counter() - self.optimizer_started

            def on_log(self, args, state, control, logs=None, **kwargs):
                append_json(output / "training.jsonl", {"step": state.global_step, **(logs or {}),
                            "elapsed_seconds": time.perf_counter() - started,
                            "stage_seconds": dict(timings.seconds)})

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % config["eval_steps"] == 0:
                    evaluate(state.global_step, "dev")
                if time.perf_counter() - started >= max_seconds:
                    control.should_training_stop = True
                return control

        args = GRPOConfig(
            output_dir=str(output / "checkpoints"), per_device_train_batch_size=2,
            gradient_accumulation_steps=16, num_generations=config["group_size"],
            max_completion_length=config["max_new_tokens"],
            chat_template_kwargs={"enable_thinking": False},
            temperature=config["temperature"], top_p=config["top_p"], top_k=config["top_k"],
            min_p=None, repetition_penalty=1.0, use_vllm=False, beta=config["beta"],
            num_iterations=1, loss_type=config["loss_type"], scale_rewards=config["scale_rewards"],
            mask_truncated_completions=False, bf16=True, gradient_checkpointing=True,
            learning_rate=config["learning_rate"], lr_scheduler_type="constant",
            max_steps=config["max_steps"], logging_steps=1, report_to="none",
            save_strategy="steps", save_steps=config["eval_steps"], save_only_model=True,
            seed=config["seed"], data_seed=config["seed"], optim="adamw_torch",
        )
        dataset = Dataset.from_list([{**p, "prompt": messages(p)} for p in splits["train"]])
        trainer = MeasuredTrainer(model=model, processing_class=tokenizer, args=args,
                                  reward_funcs=reward, train_dataset=dataset, callbacks=[Measurements()])
        trainer.train()
        step = trainer.state.global_step
        if step % config["eval_steps"]:
            evaluate(step, "dev")
        with timings.measure("checkpoint"):
            trainer.save_model(str(output / "final"))
            tokenizer.save_pretrained(output / "final")
        # Verify an ordinary HF reload, including the actual generated answers.
        before = sample(model, tokenizer, splits["dev"][:32], config, step, "reload")
        reloaded = AutoModelForCausalLM.from_pretrained(output / "final", dtype=torch.float32,
                                                       attn_implementation="sdpa").to("cuda")
        after = sample(reloaded, tokenizer, splits["dev"][:32], config, step, "reload")
        matches = all(a["token_ids"] == b["token_ids"] for a, b in zip(before, after, strict=True))
        (output / "reload.json").write_bytes(json_bytes({"step": step, "prompts": 32,
                                                       "token_ids_match": matches}))
        if not matches:
            raise RuntimeError("Checkpoint reload changed the sampled answers")
    else:
        raise ValueError("mode must be calibrate or train")

    elapsed = time.perf_counter() - started
    summary = {"status": "complete", "mode": mode, "elapsed_seconds": elapsed,
               "estimated_compute_usd": elapsed * (0.000222 + 2 * 0.0000131 + 16 * 0.00000222),
               "stage_seconds": dict(timings.seconds),
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    (output / "summary.json").write_bytes(json_bytes(summary))
    return summary
