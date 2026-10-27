"""Train Qwen3-0.6B on the reference pilot's Countdown prompt order."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile
import time

from native import Native, eos_ids, prompt_ids
from parity import prepare
from pilot import append_json, summarize
from task import verify


def training_data(experiment, data, steps, reference_run):
    config = json.loads((experiment / "config.json").read_text())
    with tarfile.open(experiment / "results.tar.gz") as archive:
        rollout_content = archive.extractfile(f"{reference_run}/rollouts.jsonl").read()
        reference_metadata = json.load(archive.extractfile(f"{reference_run}/metadata.json"))
    saved = reference_metadata["config"]
    if any(saved[key] != value for key, value in config.items() if key not in ("seed", "max_steps")):
        raise ValueError("Reference run uses a different recipe")
    if not 1 <= steps <= min(64, saved["max_steps"]):
        raise ValueError("Requested steps exceed the saved reference run")
    config.update(seed=saved["seed"], max_steps=steps)
    recipe = {"group_size": 8, "temperature": 1., "top_p": 1., "top_k": 0,
              "beta": 0., "loss_type": "grpo", "scale_rewards": "none"}
    if any(config[key] != value for key, value in recipe.items()):
        raise ValueError("The native comparison needs the frozen pilot recipe")
    manifest = json.loads((experiment / "data_manifest.json").read_text())
    if reference_metadata["data"] != manifest:
        raise ValueError("Reference run uses different splits")
    splits = {}
    for name in ("train", "dev"):
        content = (data / f"{name}.jsonl").read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["splits"][name]["sha256"]:
            raise ValueError(f"Split hash mismatch: {name}")
        splits[name] = [json.loads(line) for line in content.splitlines()]
    reference = [json.loads(line) for line in rollout_content.splitlines()]
    train = {p["id"]: p for p in splits["train"]}
    schedule = []
    for step in range(steps):
        rows = [row for row in reference if row["step"] == step]
        if len(rows) != 32:
            raise ValueError(f"Expected 32 reference rollouts at step {step}")
        ids = [row["id"] for row in rows]
        if any(len(set(ids[i:i+8])) != 1 for i in range(0, 32, 8)):
            raise ValueError("Reference groups do not contain eight responses per prompt")
        schedule.append([train[key] for key in ids[::8]])
    order = {"run": reference_run, "rollouts_sha256": hashlib.sha256(rollout_content).hexdigest()}
    return config, manifest, splits, schedule, order


def run(output, provenance, steps=64, reference_run="reference-h100-01", gpu_rate=.001097,
        max_seconds=900):
    from transformers import AutoTokenizer

    started = time.perf_counter()
    config, manifest, splits, schedule, order = training_data(
        Path("/experiment"), Path("/data"), steps, reference_run)
    model_dir, _, _, _ = prepare(output, provenance, limits={},
        reference_description="Native GRPO, centered rewards, one pass, FP16 model and FP32 Adam masters")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    metadata = json.loads((output / "metadata.json").read_text())
    metadata.update(config=config, mode="train", data=manifest,
        binary_sha256=hashlib.sha256(Path("/build/qwen_train").read_bytes()).hexdigest(),
        prompt_order=order,
        memory_method="cudaMemGetInfo after allocation, generation and training stages; sampled whole-device usage",
        timing_method="Native synchronized stage times; evaluation, checkpoint and load include Python wall time",
        rates_usd_per_second={"gpu": gpu_rate, "cpu_core": .0000131, "ram_gib": .00000222},
        cpu_cores=2, ram_gib=16)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    stages = defaultdict(float)
    settings = {"max_new_tokens": config["max_new_tokens"], "temperature": 1., "eos_ids": eos_ids(model_dir)}

    def records(puzzles, response, step, split, count):
        result = []
        for index, ids in enumerate(response["completions"]):
            puzzle = puzzles[index//count]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            verdict = verify(text, puzzle["nums"], puzzle["target"])
            result.append({**puzzle, "step": step, "split": split, "sample": index % count,
                           "text": text, "token_ids": ids, "tokens": len(ids),
                           "eos": ids[-1] in settings["eos_ids"],
                           **verdict.as_dict(), "reward": verdict.reward})
        return result

    def evaluate(engine, step, puzzles=None):
        puzzles = splits["dev"] if puzzles is None else puzzles
        rows = []
        seed = config["seed"] + 10000
        for start in range(0, len(puzzles), 32):
            batch = puzzles[start:start+32]
            response = engine.call("generate", prompts=prompt_ids(tokenizer, batch), seed=seed, **settings)
            seed = response["next_seed"]
            rows.extend(records(batch, response, step, "dev", 1))
        return rows

    with Native(model_dir, output / "native.log", config["learning_rate"]) as engine:
        stages["load"] = time.perf_counter() - started
        seed = config["seed"]
        for step in range(steps+1):
            if step % config["eval_steps"] == 0 or step == steps:
                tick = time.perf_counter()
                answers = evaluate(engine, step)
                stages["evaluation"] += time.perf_counter() - tick
                for row in answers:
                    append_json(output / "answers.jsonl", row)
                metrics = {"step": step, "split": "dev", **summarize(answers),
                           "elapsed_seconds": time.perf_counter() - started,
                           "stage_seconds": dict(stages), "sampled_device_used_bytes": engine.peak_used}
                append_json(output / "evaluations.jsonl", metrics)
                print(json.dumps(metrics), flush=True)
                if 0 < step < steps:
                    tick = time.perf_counter()
                    engine.call("save", path=str(output / f"checkpoint-{step}"))
                    stages["checkpoint"] += time.perf_counter() - tick
            if step == steps:
                break
            if time.perf_counter() - started > max_seconds:
                raise TimeoutError(f"Native training exceeded {max_seconds} seconds")
            batch = schedule[step]
            prompts = [row for row in prompt_ids(tokenizer, batch) for _ in range(8)]
            tick = time.perf_counter()
            response = engine.call("generate", prompts=prompts, seed=seed, **settings)
            stages["generation"] += time.perf_counter() - tick
            seed = response["next_seed"]
            tick = time.perf_counter()
            rollouts = records(batch, response, step, "train", 8)
            for row in rollouts:
                append_json(output / "rollouts.jsonl", row)
            stages["reward"] += time.perf_counter() - tick
            update = engine.call("update", rewards=[r["reward"] for r in rollouts], group_size=8)
            for key, value in update["stage_seconds"].items():
                stages[key] += value
            append_json(output / "training.jsonl", {"step": step+1, "loss": update["loss"],
                "gradient_norm": update["gradient_norm"], **summarize(rollouts),
                "next_seed": seed, "elapsed_seconds": time.perf_counter() - started,
                "stage_seconds": dict(stages), "sampled_device_used_bytes": engine.peak_used})
        tick = time.perf_counter()
        engine.call("save", path=str(output / "final"))
        stages["checkpoint"] += time.perf_counter() - tick
        training_peak = engine.peak_used
        before = evaluate(engine, steps, splits["dev"][:32])
    with Native(output / "final", output / "reload.log", config["learning_rate"]) as engine:
        after = evaluate(engine, steps, splits["dev"][:32])
    matches = all(a["token_ids"] == b["token_ids"] for a, b in zip(before, after, strict=True))
    (output / "reload.json").write_text(json.dumps({"step": steps, "prompts": 32,
        "token_ids_match": matches}) + "\n")
    if not matches:
        raise RuntimeError("Native checkpoint reload changed sampled answers")
    elapsed = time.perf_counter() - started
    summary = {"passed": True, "status": "complete", "mode": "train", "steps_completed": steps,
               "steps_requested": steps, "elapsed_seconds": elapsed,
               "sampled_device_used_bytes": training_peak, "stage_seconds": dict(stages),
               "estimated_compute_usd": elapsed * (gpu_rate + 2*.0000131 + 16*.00000222)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
