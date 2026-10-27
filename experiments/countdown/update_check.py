"""Check Adam updates and native checkpoints against PyTorch."""

import gc
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from parity import fixture, prepare

LIMITS = {"norm_relative": 1e-5, "weight_max_abs": 2e-7, "update_relative_l2": 0.01,
          "moment_relative": 5e-5, "moment_absolute": 1e-30}


def optimizer_check(output):
    import torch
    subprocess.run(["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "9",
                    "/build/test_qwen_optimizer", str(output / "optimizer-fixture.json")], check=True,
                    stdout=(output / "optimizer.log").open("w"), stderr=subprocess.STDOUT, timeout=60)
    data = json.loads((output / "optimizer-fixture.json").read_text())
    params = [torch.nn.Parameter(torch.tensor(data["initial_" + name], device="cuda")) for name in ("matrix", "norm")]
    optimizer = torch.optim.AdamW(params, lr=1e-6, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False, fused=False)
    checks = []
    for row in data["steps"]:
        before = [p.detach().clone() for p in params]
        for name, p in zip(("matrix", "norm"), params):
            p.grad = torch.tensor(row[name + "_grad"], device="cuda") / 256
        norm = torch.nn.utils.clip_grad_norm_(params, row["max_norm"], foreach=False).item()
        assert abs(norm-row["gradient_norm"]) <= LIMITS["norm_relative"] * max(norm, 1e-12)
        optimizer.step()
        assert torch.equal(torch.tensor(row["effective_matrix"]), torch.tensor(row["matrix"]).half().float())
        error, update = 0., 0.
        for name, p, old in zip(("matrix", "norm"), params, before):
            native = torch.tensor(row[name], device="cuda")
            assert (native-p).abs().max().item() <= LIMITS["weight_max_abs"]
            error += (native-p).square().sum().item()
            update += (p-old).square().sum().item()
            for suffix, state in (("_m", "exp_avg"), ("_v", "exp_avg_sq")):
                torch.testing.assert_close(torch.tensor(row[name + suffix], device="cuda"), optimizer.state[p][state],
                                           rtol=LIMITS["moment_relative"], atol=LIMITS["moment_absolute"])
        relative = (error / update)**.5
        assert relative <= LIMITS["update_relative_l2"]
        checks.append({"step": row["step"], "gradient_norm": norm, "update_relative_l2": relative})
    assert data["nonfinite_rejected_without_changes"]
    return checks


def check_initial(path, source, specs):
    import numpy as np
    from safetensors import safe_open
    master = np.memmap(path, dtype=np.float32, mode="r")
    expected_count = sum(spec["count"] for spec in specs if spec["pool"] == "half")
    assert len(master) == expected_count
    with safe_open(source / "model.safetensors", framework="pt") as weights:
        for spec in specs:
            if spec["pool"] == "half":
                expected = weights.get_tensor(spec["name"]).float().numpy().ravel()
                assert np.array_equal(master[spec["offset"]:spec["offset"]+spec["count"]], expected), spec["name"]


def check_updates(folder, source, specs, output):
    import numpy as np
    import torch
    from safetensors import safe_open
    with safe_open(source / "model.safetensors", framework="pt") as weights:
        params = [torch.nn.Parameter(weights.get_tensor(spec["name"]).float().cuda()) for spec in specs]
    optimizer = torch.optim.AdamW(params, lr=1e-6, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False, fused=False)
    native_steps = json.loads((folder / "steps.json").read_text())
    results = []
    for step in range(1, 4):
        gradients = {pool: np.memmap(folder / f"step{step}" / (pool+".bin"), dtype=dtype, mode="r")
                     for pool, dtype in (("half", np.float16), ("float", np.float32))}
        before = [p.detach().clone() for p in params]
        optimizer.zero_grad(set_to_none=True)
        for spec, p in zip(specs, params):
            start, end = spec["offset"], spec["offset"] + spec["count"]
            p.grad = torch.tensor(gradients[spec["pool"]][start:end].astype(np.float32).reshape(spec["shape"]), device="cuda") / 256
        norm = torch.nn.utils.clip_grad_norm_(params, 1., foreach=False).item()
        assert abs(norm-native_steps[step-1]["gradient_norm"]) <= LIMITS["norm_relative"] * norm
        optimizer.step()
        checkpoint = folder / ("final" if step == 3 else f"step{step}/model")
        tensors, error, update, maximum = [], 0., 0., 0.
        with safe_open(checkpoint / "model.safetensors", framework="pt") as weights:
            assert set(weights.keys()) == {s["name"] for s in specs}
            for spec, p, old in zip(specs, params, before):
                native = weights.get_tensor(spec["name"]).numpy()
                expected = p.detach().cpu().numpy()
                assert list(native.shape) == spec["shape"] and native.dtype == np.float32
                assert np.isfinite(native).all()
                diff, delta = native-expected, expected-old.cpu().numpy()
                tensor_error = float(np.sum(diff*diff, dtype=np.float64))
                tensor_update = float(np.sum(delta*delta, dtype=np.float64))
                max_abs = float(np.max(np.abs(diff)))
                error += tensor_error; update += tensor_update; maximum = max(maximum, max_abs)
                tensors.append({"name": spec["name"], "max_abs": max_abs, "error_squared": tensor_error, "update_squared": tensor_update})
        result = {"step": step, "gradient_norm": norm, "max_abs": maximum,
                  "update_relative_l2": (error/update)**.5, "tensors": tensors}
        result["passed"] = maximum <= LIMITS["weight_max_abs"] and result["update_relative_l2"] <= LIMITS["update_relative_l2"]
        (output / f"update-{step}.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append({k: v for k, v in result.items() if k != "tensors"})
        del before
    return results


def run(output, provenance):
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    started = time.perf_counter()
    source, inputs, specs, counts = prepare(output, provenance, LIMITS,
        reference_description="PyTorch AdamW in FP32, original source masters, identical native gradients unscaled and clipped before each update")
    source = Path(source)
    print("Checking Adam and nonfinite-gradient rejection", flush=True)
    summary = {"passed": False, "optimizer": optimizer_check(output)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    gc.collect(); torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix="qwen-update-") as tmp:
        folder = Path(tmp) / "updates"
        command = ["/build/qwen_probe", str(source), str(output/"fixture.json"), str(folder), "256", "2", "3"]
        with (output / "updates.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        print("Comparing three native updates with PyTorch", flush=True)
        check_initial(folder / "initial-master.bin", source, specs)
        summary["initial_master_exact"] = True
        summary["updates"] = check_updates(folder, source, specs, output)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        gc.collect(); torch.cuda.empty_cache()
        print("Checking native and Transformers checkpoint reloads", flush=True)
        final = folder / "final"
        reloaded = Path(tmp) / "reload"
        command = ["/build/qwen_probe", str(final), str(output/"fixture.json"), str(reloaded), "256", "2", "0"]
        with (output / "reload.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        check_initial(reloaded / "initial-master.bin", final, specs)
        with safe_open(final / "model.safetensors", framework="pt") as left, safe_open(reloaded / "final/model.safetensors", framework="pt") as right:
            assert set(left.keys()) == set(right.keys())
            assert all(torch.equal(left.get_tensor(name), right.get_tensor(name)) for name in left.keys())
        summary["native_reload_exact"] = True
        model = AutoModelForCausalLM.from_pretrained(final, dtype=torch.float32, attn_implementation="eager").cuda().eval()
        with safe_open(final / "model.safetensors", framework="pt") as weights:
            assert all(torch.equal(p.detach().cpu(), weights.get_tensor(name)) for name, p in model.named_parameters())
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        assert fixture(AutoTokenizer.from_pretrained(final)) == inputs
        ids = torch.tensor(inputs["tokens"], device="cuda").reshape(2, 193)
        with torch.no_grad():
            assert torch.isfinite(model(input_ids=ids, use_cache=False).logits).all()
        summary["transformers_reload_exact"] = True
        shutil.copytree(final, output / "final")
    summary["passed"] = all(row["passed"] for row in summary["updates"])
    summary["seconds"] = time.perf_counter() - started
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
