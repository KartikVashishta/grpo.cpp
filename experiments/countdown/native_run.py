"""Build and check the native Qwen model on Modal."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent if modal.is_local() else Path("/experiment")
ROOT = HERE.parent.parent if modal.is_local() else Path("/src")
app = modal.App("grpo-native")
volume = modal.Volume.from_name("grpo-countdown")
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
         .apt_install("cmake", "ninja-build", "g++", "patch", "ca-certificates")
         .pip_install_from_requirements(str(HERE / "requirements.lock"))
         .env({"HF_HOME": "/artifacts/hf", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_file(ROOT / "CMakeLists.txt", "/src/CMakeLists.txt", copy=True))
for directory in ("cmake", "include", "src", "tests", "bench", "examples"):
    image = image.add_local_dir(ROOT / directory, "/src/" + directory,
                                ignore=["__pycache__"], copy=True)
image = (image.run_commands(
    "cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGRPO_USE_CUDA=ON -DGRPO_BUILD_QWEN=ON -DCMAKE_CUDA_ARCHITECTURES=90",
    "cmake --build /build --target qwen_probe bench_grpo_loss_cuda -j2")
    .add_local_dir(HERE, "/experiment", ignore=["__pycache__"]))


@app.function(image=image, timeout=120)
def build_check():
    import shutil
    import sys
    sys.path.insert(0, "/experiment")
    from parity import fixture
    from transformers import AutoTokenizer
    assert shutil.which("compute-sanitizer")
    assert Path("/build/qwen_probe").is_file()
    config = json.loads(Path("/experiment/config.json").read_text())
    inputs = fixture(AutoTokenizer.from_pretrained(config["model"], revision=config["model_revision"]))
    return {"nvcc": subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True),
            "prompt_lengths": inputs["prompt_lengths"], "completion_lengths": inputs["completion_lengths"]}


@app.function(image=image, gpu="H100", cpu=(2, 2), memory=(16384, 16384),
              timeout=1200, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": volume})
def execute(run_id, provenance):
    import sys
    sys.path.insert(0, "/experiment")
    from parity import run
    output = Path("/artifacts") / run_id
    output.mkdir()
    started = time.perf_counter()
    try:
        run(output, provenance)
    except Exception as error:
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
def main(run_id: str = "", build_only: bool = False):
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
        files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "files_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
    }
    result = execute.remote(run_id, provenance)
    output.mkdir(parents=True)
    for name, content in result.items():
        (output / name).write_text(content)
    print(result["summary.json"])
    if not json.loads(result["summary.json"])["passed"]:
        raise SystemExit(1)
