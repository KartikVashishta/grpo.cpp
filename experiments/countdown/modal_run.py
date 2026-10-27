"""Run the Countdown pilot on Modal."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent if modal.is_local() else Path("/experiment")
ROOT = HERE.parent.parent
DATA = ROOT / "runs/countdown/data" if modal.is_local() else Path("/data")
app = modal.App("grpo-countdown")
volume = modal.Volume.from_name("grpo-countdown", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements(str(HERE / "requirements.lock"))
         .env({"HF_HOME": "/artifacts/hf", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir(HERE, "/experiment", ignore=["__pycache__"])
         .add_local_dir(DATA, "/data", ignore=["source.parquet"]))


@app.function(image=image, gpu="L4", cpu=(2, 2), memory=(16384, 16384), timeout=3600,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={"/artifacts": volume})
def execute(config, run_id, mode, provenance):
    import sys
    sys.path.insert(0, "/experiment")
    from pilot import run

    output = Path("/artifacts") / run_id
    if output.exists():
        raise ValueError(f"Remote run already exists: {run_id}")
    started = time.perf_counter()
    try:
        run(config, Path("/data"), output, mode, provenance,
            max_seconds=900 if config["gpu"] == "H100" else 3000)
    finally:
        if output.exists():
            volume.commit()
            elapsed = time.perf_counter() - started
            (output / "container.json").write_text(json.dumps({
                "function_seconds": elapsed,
                "estimated_compute_usd": elapsed * (config["gpu_rate_usd_per_second"] + 2 * 0.0000131 + 16 * 0.00000222),
                "note": "Includes artifact commit; excludes the final small timing-file commit, image build, startup before function entry, storage and network; not an invoice."
            }, indent=2) + "\n")
            volume.commit()
    return {p.name: p.read_text() for p in output.iterdir()
            if p.is_file() and p.suffix in (".json", ".jsonl")}


@app.local_entrypoint()
def main(run_id: str, mode: str = "calibrate", steps: int = 64, gpu: str = "L4"):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_id):
        raise ValueError("Use a lowercase run ID containing only letters, numbers, '-' or '_'")
    if mode not in ("calibrate", "train") or not 1 <= steps <= 256:
        raise ValueError("Use calibrate/train and 1–256 steps")
    rates = {"L4": 0.000222, "H100": 0.001097}
    if gpu not in rates:
        raise ValueError("Supported measured configurations: L4 or H100")
    output = ROOT / "runs/countdown" / run_id
    if output.exists():
        raise ValueError(f"Run already exists: {output}")
    frozen = json.loads((HERE / "data_manifest.json").read_text())
    if json.loads((ROOT / "runs/countdown/data/manifest.json").read_text()) != frozen:
        raise ValueError("Prepare the frozen splits before launching compute")
    config = json.loads((HERE / "config.json").read_text())
    config["max_steps"] = steps
    config["gpu"] = gpu
    config["gpu_rate_usd_per_second"] = rates[gpu]
    tracked = sorted(p for p in HERE.iterdir() if p.is_file())
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
    }
    function = execute if gpu == "L4" else execute.with_options(gpu=gpu, timeout=1200)
    result = function.remote(config, run_id, mode, provenance)
    output.mkdir(parents=True)
    for name, content in result.items():
        (output / name).write_text(content)
    print(f"Saved measurements to {output}")
    print(result["summary.json"])
