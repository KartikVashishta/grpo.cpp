"""Build and run the native Qwen model on Modal."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent if modal.is_local() else Path("/experiment")
ROOT = HERE.parent.parent if modal.is_local() else Path("/src")
DATA = ROOT / "runs/countdown/data" if modal.is_local() else Path("/data")
app = modal.App("grpo-native")
volume = modal.Volume.from_name("grpo-countdown")
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
         .apt_install("cmake", "ninja-build", "g++", "patch", "ca-certificates")
         .pip_install_from_requirements(str(HERE / "requirements.lock"))
         .env({"HF_HOME": "/artifacts/hf", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_file(ROOT / "CMakeLists.txt", "/src/CMakeLists.txt", copy=True))
for directory in ("cmake", "include", "src", "tests", "bench", "examples"):
    image = image.add_local_dir(ROOT / directory, "/src/" + directory,
                                ignore=["__pycache__", ".DS_Store"], copy=True)
image = (image.run_commands(
    "cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGRPO_USE_CUDA=ON -DGRPO_BUILD_QWEN=ON -DCMAKE_CUDA_ARCHITECTURES=90",
    "cmake --build /build --target qwen_train qwen_probe test_qwen_sampler test_qwen_embedding test_qwen_optimizer bench_grpo_loss_cuda -j2")
    .add_local_dir(HERE, "/experiment", ignore=["__pycache__", ".DS_Store"])
    .add_local_dir(DATA, "/data", ignore=["source.parquet", ".DS_Store"]))


@app.function(image=image, timeout=120)
def build_check():
    import shutil
    import sys
    sys.path.insert(0, "/experiment")
    from parity import fixture
    from native import prompt_ids
    from transformers import AutoTokenizer
    assert shutil.which("compute-sanitizer")
    assert "--log-file" in subprocess.check_output(["compute-sanitizer", "--help"], text=True)
    assert Path("/build/qwen_probe").is_file()
    assert Path("/build/test_qwen_embedding").is_file()
    assert Path("/build/test_qwen_optimizer").is_file()
    assert Path("/build/qwen_train").is_file()
    assert Path("/build/test_qwen_sampler").is_file()
    config = json.loads(Path("/experiment/config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["model_revision"])
    inputs = fixture(tokenizer)
    prompts = prompt_ids(tokenizer, [{"nums": [43, 44, 58], "target": 57},
                                     {"nums": [17, 57, 81], "target": 41}])
    assert all(row == inputs["tokens"][i*193:i*193+len(row)] for i, row in enumerate(prompts))
    return {"nvcc": subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True),
            "prompt_lengths": inputs["prompt_lengths"], "completion_lengths": inputs["completion_lengths"]}


@app.function(image=image, gpu="H100", cpu=(2, 2), memory=(16384, 16384),
              timeout=1200, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": volume})
def execute(run_id, provenance, mode, steps):
    import sys
    sys.path.insert(0, "/experiment")
    if mode == "parity":
        from parity import run
    elif mode == "update":
        from update_check import run
    elif mode == "generation":
        from generation_check import run
    else:
        from native_train import run
    output = Path("/artifacts") / run_id
    output.mkdir()
    started = time.perf_counter()
    try:
        if mode == "train":
            run(output, provenance, steps)
        else:
            run(output, provenance)
    except Exception as error:
        import traceback
        (output / "error.txt").write_text(traceback.format_exc())
        path = output / "summary.json"
        summary = json.loads(path.read_text()) if path.exists() else {}
        summary.update(passed=False, error=f"{type(error).__name__}: {error}")
        path.write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        volume.commit()
        elapsed = time.perf_counter() - started
        (output / "container.json").write_text(json.dumps({
            "function_seconds": elapsed,
            "estimated_compute_usd": elapsed * (0.001097 + 2*0.0000131 + 16*0.00000222),
            "note": "Function time including artifact commit; excludes startup, image build, storage and network."
        }, indent=2) + "\n")
        volume.commit()
    return {p.name: p.read_text() for p in output.iterdir() if p.is_file()}


@app.local_entrypoint()
def main(run_id: str = "", build_only: bool = False, mode: str = "parity", steps: int = 64):
    if mode not in ("parity", "update", "generation", "train") or not 1 <= steps <= 64:
        raise ValueError("Use parity, update, generation or train, with 1 to 64 steps")
    if build_only:
        print(build_check.remote())
        return
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_id):
        raise ValueError("Use a lowercase run ID")
    output = ROOT / "runs/countdown" / run_id
    if output.exists():
        raise ValueError(f"Run already exists: {output}")
    files = [ROOT / "CMakeLists.txt"]
    for directory in ("cmake", "include", "src", "tests", "bench", "examples", "experiments/countdown"):
        files.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store")
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "files_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
    }
    result = execute.remote(run_id, provenance, mode, steps)
    output.mkdir(parents=True)
    for name, content in result.items():
        (output / name).write_text(content)
    print(result["summary.json"])
    if not json.loads(result["summary.json"])["passed"]:
        raise SystemExit(1)
