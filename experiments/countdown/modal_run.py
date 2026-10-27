"""Run one bounded L4 job; credentials come from the local Modal profile."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
app = modal.App("grpo-countdown")
volume = modal.Volume.from_name("grpo-countdown", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements(str(HERE / "requirements.txt"))
         .env({"HF_HOME": "/artifacts/hf", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir(HERE, "/experiment", ignore=["__pycache__"])
         .add_local_dir(ROOT / "runs/countdown/data", "/data", ignore=["source.parquet"]))


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
        run(config, Path("/data"), output, mode, provenance)
    finally:
        if output.exists():
            elapsed = time.perf_counter() - started
            (output / "container.json").write_text(json.dumps({
                "function_seconds": elapsed,
                "estimated_compute_usd": elapsed * (0.000222 + 2 * 0.0000131 + 16 * 0.00000222),
                "note": "Estimate excludes image build, startup before function entry, storage and network; not an invoice."
            }, indent=2) + "\n")
            volume.commit()
    return {p.name: p.read_text() for p in output.iterdir()
            if p.is_file() and p.suffix in (".json", ".jsonl")}


@app.local_entrypoint()
def main(run_id: str, mode: str = "calibrate", steps: int = 64):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_id):
        raise ValueError("Use a lowercase run ID containing only letters, numbers, '-' or '_'")
    if mode not in ("calibrate", "train") or not 1 <= steps <= 256:
        raise ValueError("Use calibrate/train and 1–256 steps")
    output = ROOT / "runs/countdown" / run_id
    if output.exists():
        raise ValueError(f"Run already exists: {output}")
    frozen = json.loads((HERE / "data_manifest.json").read_text())
    if json.loads((ROOT / "runs/countdown/data/manifest.json").read_text()) != frozen:
        raise ValueError("Prepare the frozen splits before launching compute")
    config = json.loads((HERE / "config.json").read_text())
    config["max_steps"] = steps
    tracked = sorted(p for p in HERE.iterdir() if p.is_file())
    provenance = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
    }
    result = execute.remote(config, run_id, mode, provenance)
    output.mkdir(parents=True)
    for name, content in result.items():
        (output / name).write_text(content)
    print(f"Saved measurements to {output}")
    print(result["summary.json"])
