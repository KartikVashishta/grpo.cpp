"""Compare the native forward pass and parameter gradients with PyTorch."""

import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time

from task import messages


LIMITS = {
    "logp_max_abs": 0.05, "logp_rms": 0.015,
    "gradient_relative_l2": 0.03, "gradient_cosine": 0.999,
    "tensor_relative_l2": 0.10, "tensor_cosine": 0.99,
    "near_zero_reference_rms": 1e-9, "near_zero_max_abs": 1e-8,
    "microbatch_relative_l2": 0.03,
    "padding_logp_max_abs": 1e-5, "padding_gradient_relative_l2": 1e-5,
}


def fixture(tokenizer):
    import numpy as np
    tokens = np.full((2, 193), tokenizer.pad_token_id, dtype=np.int32)
    targets = np.zeros_like(tokens)
    upstream = np.zeros((2, 193), dtype=np.float32)
    lengths, prompt_lengths = [], []
    examples = [([43, 44, 58], 57, "(58 - 44) + 43"),
                ([17, 57, 81], 41, "81 - 57 + 17")]
    for b, (numbers, target, answer) in enumerate(examples):
        text = tokenizer.apply_chat_template(messages({"nums": numbers, "target": target}),
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        prompt = tokenizer.encode(text, add_special_tokens=False)
        completion = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        full = prompt + completion
        n = len(full) - 1
        assert n < 193
        tokens[b, :n], targets[b, :n] = full[:-1], full[1:]
        upstream[b, len(prompt)-1:n] = (0.5 if b == 0 else -0.5) / (32 * len(completion))
        assert targets[b, n-1] == tokenizer.eos_token_id
        lengths.append(n)
        prompt_lengths.append(len(prompt))
    active = np.flatnonzero(upstream.ravel())
    assert active.min() < 256 <= active.max()
    assert any(i % 193 >= 128 for i in active)
    return {"batch": 2, "sequence": 193, "tokens": tokens.ravel().tolist(),
            "targets": targets.ravel().tolist(), "upstream": upstream.ravel().tolist(),
            "lengths": lengths, "prompt_lengths": prompt_lengths,
            "completion_lengths": [n+1-p for n, p in zip(lengths, prompt_lengths)]}


def layout(config):
    h, q, kv = config.hidden_size, config.num_attention_heads * config.head_dim, config.num_key_value_heads * config.head_dim
    inter = config.intermediate_size
    specs = []
    offsets = {"half": 0, "float": 0}

    def add(name, shape, pool):
        count = 1
        for dim in shape:
            count *= dim
        specs.append({"name": name, "shape": shape, "pool": pool, "offset": offsets[pool], "count": count})
        offsets[pool] += count

    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}."
        for name, shape in [("self_attn.q_proj", [q, h]), ("self_attn.k_proj", [kv, h]),
                            ("self_attn.v_proj", [kv, h]), ("self_attn.o_proj", [h, q]),
                            ("mlp.gate_proj", [inter, h]), ("mlp.up_proj", [inter, h]),
                            ("mlp.down_proj", [h, inter])]:
            add(prefix + name + ".weight", shape, "half")
        for name, size in [("input_layernorm", h), ("self_attn.q_norm", config.head_dim),
                           ("self_attn.k_norm", config.head_dim), ("post_attention_layernorm", h)]:
            add(prefix + name + ".weight", [size], "float")
    add("model.embed_tokens.weight", [config.vocab_size, h], "half")
    add("model.norm.weight", [h], "float")
    return specs, offsets


def reference(model_dir, inputs, specs, counts, output, autocast=False):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    output.mkdir()
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32,
                                               attn_implementation="eager").cuda().eval()
    parameters = dict(model.named_parameters())
    assert set(parameters) == {s["name"] for s in specs}
    with torch.no_grad():
        for spec in specs:
            p = parameters[spec["name"]]
            assert list(p.shape) == spec["shape"]
            if spec["pool"] == "half":
                p.copy_(p.half().float())
    ids = torch.tensor(inputs["tokens"], device="cuda").reshape(2, 193)
    targets = torch.tensor(inputs["targets"], device="cuda").reshape(2, 193)
    upstream = torch.tensor(inputs["upstream"], device="cuda").reshape(2, 193)
    context = torch.autocast("cuda", dtype=torch.float16) if autocast else contextlib.nullcontext()
    with context:
        logits = model(input_ids=ids, use_cache=False).logits.float()
        logp = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
        loss = (logp * upstream).sum()
    scale = 256 if autocast else 1
    (loss * scale).backward()
    (output / "logp.json").write_text(json.dumps(logp.detach().cpu().flatten().tolist()))
    for pool, count in counts.items():
        data = np.memmap(output / (pool + ".bin"), dtype=np.float32, mode="w+", shape=(count,))
        for spec in specs:
            if spec["pool"] == pool:
                grad = parameters[spec["name"]].grad
                assert grad is not None and torch.isfinite(grad).all()
                data[spec["offset"]:spec["offset"]+spec["count"]] = grad.detach().cpu().numpy().ravel() / scale
        data.flush()
        del data
    del model, parameters, loss, logits, logp, grad, p
    torch.cuda.empty_cache()


def compare(left, right, specs, counts, left_scale=1, right_scale=1,
            left_native=True, right_native=False):
    import numpy as np
    pools = []
    for path, native in [(left, left_native), (right, right_native)]:
        group = {}
        for pool, count in counts.items():
            dtype = np.float16 if native and pool == "half" else np.float32
            assert (path / (pool + ".bin")).stat().st_size == count * np.dtype(dtype).itemsize
            group[pool] = np.memmap(path / (pool + ".bin"), mode="r", dtype=dtype)
        pools.append(group)
    tensors, total = [], np.zeros(4, dtype=np.float64)
    for spec in specs:
        sums = np.zeros(4, dtype=np.float64)
        maximum, zeros, finite = 0., 0, True
        start, end = spec["offset"], spec["offset"] + spec["count"]
        for offset in range(start, end, 1_000_000):
            a = pools[0][spec["pool"]][offset:min(end, offset+1_000_000)].astype(np.float64) / left_scale
            b = pools[1][spec["pool"]][offset:min(end, offset+1_000_000)].astype(np.float64) / right_scale
            finite = finite and bool(np.isfinite(a).all() and np.isfinite(b).all())
            d = a-b
            sums += [np.sum(a*a), np.sum(b*b), np.sum(d*d), np.sum(a*b)]
            maximum = max(maximum, float(np.max(np.abs(d))))
            zeros += int(np.count_nonzero((a == 0) & (b != 0)))
        total += sums
        norm_a, norm_b, error, dot = sums
        relative = float(np.sqrt(error / norm_b)) if norm_b else (0. if error == 0 else None)
        cosine = float(dot / np.sqrt(norm_a * norm_b)) if norm_a * norm_b else (1. if error == 0 else 0.)
        rms = float(np.sqrt(norm_b / spec["count"]))
        passed = (maximum <= LIMITS["near_zero_max_abs"] if rms <= LIMITS["near_zero_reference_rms"]
                  else relative is not None and relative <= LIMITS["tensor_relative_l2"] and cosine >= LIMITS["tensor_cosine"])
        tensors.append({"name": spec["name"], "relative_l2": relative, "cosine": cosine,
                        "reference_norm": float(np.sqrt(norm_b)), "reference_rms": rms,
                        "max_abs": maximum, "lost_nonzero": zeros, "finite": finite,
                        "passed": bool(finite and passed)})
    a, b, d, dot = total
    return {"relative_l2": float(np.sqrt(d/b)) if b else None,
            "cosine": float(dot/np.sqrt(a*b)) if a*b else None,
            "finite": all(t["finite"] for t in tensors),
            "max_abs": max(t["max_abs"] for t in tensors), "tensors": tensors}


def check_loader(model_dir, fixture_path, tmp):
    import torch
    from safetensors.torch import load_file, save_file

    folder = tmp / "weights"
    folder.mkdir()
    shutil.copy(Path(model_dir) / "config.json", folder / "config.json")
    name = "model.layers.0.self_attn.q_proj.weight"
    tests = [("missing", {}, "Missing tensor"),
             ("shape", {name: torch.zeros(1024, 2048)}, "Wrong tensor shape"),
             ("dtype", {name: torch.zeros(2048, 1024, dtype=torch.int32)}, "Unsupported weight dtype"),
             ("nonfinite", {name: torch.full((2048, 1024), float("inf"))}, "Nonfinite FP16 weight")]
    results = {}
    for label, tensors, expected in tests:
        save_file(tensors, folder / "model.safetensors")
        result = subprocess.run(["/build/qwen_probe", str(folder), str(fixture_path), str(tmp/label), "1", "2"],
                                capture_output=True, text=True, timeout=30)
        results[label] = {"returncode": result.returncode, "stderr": result.stderr}
        if result.returncode == 0 or expected not in result.stderr:
            raise AssertionError(f"Loader accepted {label}, or failed for a different reason: {result.stderr}")
    header = json.dumps({name: {"dtype": "F32", "shape": [2048, 1024], "data_offsets": [0, 16]}}).encode()
    for label, data, expected in [("truncated", b"short", "Truncated safetensors file"),
                                  ("offsets", struct.pack("<Q", len(header)) + header, "Invalid tensor offsets")]:
        (folder / "model.safetensors").write_bytes(data)
        result = subprocess.run(["/build/qwen_probe", str(folder), str(fixture_path), str(tmp/label), "1", "2"],
                                capture_output=True, text=True, timeout=30)
        results[label] = {"returncode": result.returncode, "stderr": result.stderr}
        if result.returncode == 0 or expected not in result.stderr:
            raise AssertionError(f"Loader did not reject {label}: {result.stderr}")
    tensors = {name: tensor.float() for name, tensor in load_file(Path(model_dir) / "model.safetensors").items()}
    save_file(tensors, folder / "model.safetensors")
    return folder, results


def run(output, provenance):
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    config = json.loads(Path("/experiment/config.json").read_text())
    model_dir = snapshot_download(config["model"], revision=config["model_revision"],
                                  allow_patterns=["*.json", "*.safetensors"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    inputs = fixture(tokenizer)
    specs, counts = layout(AutoConfig.from_pretrained(model_dir))
    (output / "fixture.json").write_text(json.dumps(inputs, indent=2) + "\n")
    runtime_hashes = {}
    for name in provenance["files_sha256"]:
        path = Path("/experiment") / Path(name).name if name.startswith("experiments/countdown/") else Path("/src") / name
        runtime_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert runtime_hashes == provenance["files_sha256"]
    (output / "metadata.json").write_text(json.dumps({
        "source": provenance, "runtime_files_sha256": runtime_hashes, "limits": LIMITS,
        "model": config["model"], "revision": config["model_revision"],
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "parameters": specs,
        "binary_sha256": hashlib.sha256(Path("/build/qwen_probe").read_bytes()).hexdigest(),
        "nvcc": subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True),
        "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
        "reference": "FP32 eager, TF32 off; projections and embedding rounded to FP16, norms kept FP32"
    }, indent=2) + "\n")
    subprocess.run(["/build/bench_grpo_loss_cuda", "--check-only"], check=True,
                   stdout=(output / "loss-check.txt").open("w"), stderr=subprocess.STDOUT, timeout=60)
    summary = {"passed": False, "cases": {}}
    with tempfile.TemporaryDirectory(prefix="qwen-parity-") as tmp:
        tmp = Path(tmp)
        fp32_model, loader_checks = check_loader(model_dir, output / "fixture.json", tmp)
        (output / "loader.json").write_text(json.dumps(loader_checks, indent=2) + "\n")
        reference(model_dir, inputs, specs, counts, tmp / "reference")
        reference(model_dir, inputs, specs, counts, tmp / "torch_autocast", autocast=True)
        (output / "torch-autocast.json").write_text(json.dumps(compare(
            tmp / "torch_autocast", tmp / "reference", specs, counts, left_native=False), indent=2) + "\n")
        active = np.asarray(inputs["upstream"]) != 0
        reference_logp = np.array(json.loads((tmp / "reference/logp.json").read_text()))
        for name, scale, microbatch in [("unscaled", 1, 2), ("scaled", 256, 2),
                                         ("repeat", 256, 2),
                                         ("microbatch", 256, 1), ("padding", 256, 2),
                                         ("fp32_weights", 256, 2), ("zero", 256, 2)]:
            if time.perf_counter() - started > 900:
                raise TimeoutError("Native check exceeded 15 minutes")
            case = dict(inputs)
            if name == "zero":
                case["upstream"] = [0.] * 386
            if name == "padding":
                ids = np.array(inputs["tokens"]).reshape(2, 193)
                for b, length in enumerate(inputs["lengths"]):
                    ids[b, length:] = 42
                case["tokens"] = ids.ravel().tolist()
            fixture_path = tmp / (name + ".json")
            fixture_path.write_text(json.dumps(case))
            weights = str(fp32_model) if name == "fp32_weights" else model_dir
            command = ["/build/qwen_probe", weights, str(fixture_path), str(tmp/name), str(scale), str(microbatch)]
            if name == "scaled":
                command = ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "9"] + command
            with (output / (name + ".log")).open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            logp = np.array(json.loads((tmp/name/"logp.json").read_text()))
            error = (logp-reference_logp)[active]
            result = {"logp_max_abs": float(np.max(np.abs(error))), "logp_rms": float(np.sqrt(np.mean(error**2)))}
            if name == "zero":
                result["exact_zero"] = all(not np.any(np.memmap(tmp/name/(pool+".bin"), mode="r",
                    dtype=np.float16 if pool == "half" else np.float32)) for pool in counts)
                result["passed"] = result["exact_zero"]
            else:
                other = tmp / ("scaled" if name in ("repeat", "padding", "microbatch", "fp32_weights") else "reference")
                check = compare(tmp/name, other, specs, counts, left_scale=scale,
                                right_scale=256 if other.name == "scaled" else 1,
                                right_native=other.name == "scaled")
                (output / (name + ".json")).write_text(json.dumps(check, indent=2) + "\n")
                result.update({k: v for k, v in check.items() if k != "tensors"})
                if name in ("repeat", "padding", "fp32_weights"):
                    base_logp = np.array(json.loads((other/"logp.json").read_text()))
                    result["baseline_logp_max_abs"] = float(np.max(np.abs((logp-base_logp)[active])))
                    result["passed"] = check["finite"] and check["relative_l2"] <= LIMITS["padding_gradient_relative_l2"] and result["baseline_logp_max_abs"] <= LIMITS["padding_logp_max_abs"]
                elif name == "microbatch":
                    result["passed"] = check["finite"] and check["relative_l2"] <= LIMITS["microbatch_relative_l2"]
                else:
                    result["failed_tensors"] = [t["name"] for t in check["tensors"] if not t["passed"]]
                    result["passed"] = (check["finite"] and not result["failed_tensors"]
                        and check["relative_l2"] <= LIMITS["gradient_relative_l2"]
                        and check["cosine"] >= LIMITS["gradient_cosine"]
                        and result["logp_max_abs"] <= LIMITS["logp_max_abs"]
                        and result["logp_rms"] <= LIMITS["logp_rms"])
            summary["cases"][name] = result
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            if name != "scaled":
                shutil.rmtree(tmp/name)
        summary["passed"] = all(summary["cases"][name]["passed"] for name in ("scaled", "repeat", "microbatch", "padding", "fp32_weights", "zero"))
    summary["seconds"] = time.perf_counter() - started
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
