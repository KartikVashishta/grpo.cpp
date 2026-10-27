"""Evaluate the fixed 64-update checkpoints on the untouched test split."""

import gc
import hashlib
import json
from pathlib import Path
import tarfile
import time

from native import Native, eos_ids, prompt_ids
from parity import prepare
from pilot import append_json, sample, summarize
from task import verify


def checkpoints(experiment, names):
    if not names or len(set(names)) != len(names):
        raise ValueError("Choose each checkpoint run once")
    config = json.loads((experiment / "config.json").read_text())
    manifest = json.loads((experiment / "data_manifest.json").read_text())
    result = []
    with tarfile.open(experiment / "results.tar.gz") as archive:
        for name in names:
            metadata = json.load(archive.extractfile(f"{name}/metadata.json"))
            summary = json.load(archive.extractfile(f"{name}/summary.json"))
            if summary["status"] != "complete" or summary["steps_completed"] != 64:
                raise ValueError(f"Expected a completed 64-update run: {name}")
            if any(metadata["config"][k] != v for k, v in config.items() if k != "seed"):
                raise ValueError(f"Different training recipe: {name}")
            if metadata["data"] != manifest:
                raise ValueError(f"Different data splits: {name}")
            result.append({"run": name, "seed": metadata["config"]["seed"],
                           "implementation": "native" if "prompt_order" in metadata else "trl",
                           "training_source": metadata["source"]})
    return config, manifest, result


def run(output, provenance, names):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    config, manifest, selected = checkpoints(Path("/experiment"), names)
    for row in selected:
        path = Path("/artifacts") / row["run"] / "final/model.safetensors"
        with path.open("rb") as file:
            row["weights_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
    data = Path("/data/test.jsonl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["splits"]["test"]["sha256"]
    puzzles = [json.loads(line) for line in data.splitlines()]
    assert len(puzzles) == 512 and len({p["id"] for p in puzzles}) == 512
    model_dir, _, _, _ = prepare(output, provenance, limits={},
        reference_description="Final test, fixed 64-update checkpoints; same sampling as development")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    metadata = json.loads((output / "metadata.json").read_text())
    metadata.update(config=config, data=manifest, checkpoints=selected,
                    selection="Final checkpoint at 64 updates for every run; no test-driven selection")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    baselines = {}
    results = []
    for selected_run in selected:
        implementation, seed = selected_run["implementation"], selected_run["seed"]
        key = (implementation, seed)
        labels = ([f"{implementation}-base-{seed}"] if key not in baselines else []) + [selected_run["run"]]
        for label in labels:
            initial = label != selected_run["run"]
            weights = model_dir if initial else Path("/artifacts") / label / "final"
            step = 0 if initial else 64
            config["seed"] = seed
            tick = time.perf_counter()
            if implementation == "native":
                records = []
                sampling_seed = seed + 10000
                eos = eos_ids(weights)
                with Native(weights, output / f"{label}.log") as engine:
                    for start in range(0, 512, 32):
                        batch = puzzles[start:start+32]
                        response = engine.call("generate", prompts=prompt_ids(tokenizer, batch),
                            seed=sampling_seed, eos_ids=eos, max_new_tokens=64, temperature=1.)
                        sampling_seed = response["next_seed"]
                        for puzzle, ids in zip(batch, response["completions"], strict=True):
                            text = tokenizer.decode(ids, skip_special_tokens=True)
                            verdict = verify(text, puzzle["nums"], puzzle["target"])
                            records.append({**puzzle, "step": step, "split": "test", "sample": 0,
                                "text": text, "token_ids": ids, "tokens": len(ids), "eos": ids[-1] in eos,
                                **verdict.as_dict(), "reward": verdict.reward})
            else:
                model = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.float32,
                                                            attn_implementation="sdpa").cuda()
                records = sample(model, tokenizer, puzzles, config, step, "test")
                del model
                gc.collect()
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            assert len(records) == 512 and [r["id"] for r in records] == [p["id"] for p in puzzles]
            for record in records:
                append_json(output / "answers.jsonl", {"run": label, **record})
            result = {"run": label, "implementation": implementation, "seed": seed, "step": step,
                      "seconds": time.perf_counter()-tick, **summarize(records)}
            if initial:
                baselines[key] = records
            else:
                pairs = list(zip(baselines[key], records, strict=True))
                result["gains"] = sum(a["reward"] == 0 and b["reward"] == 1 for a, b in pairs)
                result["regressions"] = sum(a["reward"] == 1 and b["reward"] == 0 for a, b in pairs)
            results.append(result)
            append_json(output / "evaluations.jsonl", result)
            print(json.dumps(result), flush=True)
    summary = {"passed": True, "status": "complete", "mode": "test", "results": results,
               "seconds": time.perf_counter()-started}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
