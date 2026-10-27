"""Download the pinned source and freeze the four Countdown splits."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from task import json_bytes, make_splits


def prepare(config, output):
    import pyarrow.parquet as pq

    output.mkdir(parents=True, exist_ok=True)
    source = output / "source.parquet"
    url = (f"https://huggingface.co/datasets/{config['dataset']}/resolve/"
           f"{config['dataset_revision']}/data/train-00000-of-00001.parquet")
    if not source.exists():
        temporary = source.with_suffix(".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(source)
    rows = pq.read_table(source).to_pylist()
    splits, distinct = make_splits(rows, config["sizes"], config["split_seed"])
    manifest = {
        "dataset": config["dataset"], "revision": config["dataset_revision"],
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": len(rows), "distinct_three_number_puzzles": distinct,
        "split_seed": config["split_seed"], "splits": {},
    }
    for name, puzzles in splits.items():
        content = b"".join((json.dumps(p, sort_keys=True) + "\n").encode() for p in puzzles)
        destination = output / f"{name}.jsonl"
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"Refusing to change frozen split {destination}; use a new directory")
        destination.write_bytes(content)
        manifest["splits"][name] = {"count": len(puzzles), "sha256": hashlib.sha256(content).hexdigest()}
    manifest_path = output / "manifest.json"
    content = json_bytes(manifest)
    if manifest_path.exists() and manifest_path.read_bytes() != content:
        raise ValueError("Frozen manifest changed; use a new directory")
    manifest_path.write_bytes(content)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/countdown/data"))
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    print(json.dumps(prepare(config, args.output), indent=2))
