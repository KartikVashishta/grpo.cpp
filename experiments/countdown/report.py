"""Plot a Countdown run and browse its saved answers."""

import argparse
import base64
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import re


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def pair_answers(metrics, answers, runs):
    """Join baseline and trained answers by puzzle, within each seed and trainer."""
    paired = []
    for after in metrics:
        if after["step"] == 0:
            continue
        before = next(row for row in metrics if row["step"] == 0
                      and (row["implementation"], row["seed"]) == (after["implementation"], after["seed"]))
        rows = []
        for metric in (before, after):
            selected = [a for a in answers if a["run"] == metric["run"] and a["step"] == metric["step"]]
            by_id = {a["id"]: a for a in selected}
            if len(by_id) != len(selected):
                raise ValueError(f"Duplicate final-test puzzle in {metric['run']}")
            rows.append(by_id)
        if not rows[0] or rows[0].keys() != rows[1].keys():
            raise ValueError(f"Unpaired final-test puzzles in {after['run']}")
        puzzles = []
        for key, old in rows[0].items():
            new = rows[1][key]
            if (old["nums"], old["target"]) != (new["nums"], new["target"]):
                raise ValueError(f"Puzzle changed: {key}")
            outcome = {(False, True): "improved", (True, False): "regressed",
                       (False, False): "still_wrong", (True, True): "still_correct"}[
                           bool(old["reward"]), bool(new["reward"])]
            puzzles.append({"id": key, "nums": old["nums"], "target": old["target"],
                            "before": {k: old[k] for k in ("text", "status", "value", "reward")},
                            "after": {k: new[k] for k in ("text", "status", "value", "reward")}, "outcome": outcome})
        run = next(r for r in runs if r["name"] == after["run"])
        paired.append({"name": after["run"], "seed": after["seed"], "step": after["step"],
                       "label": f"{'grpo.cpp' if after['implementation'] == 'native' else 'TRL'} · seed {after['seed']}",
                       "seconds": run["container"]["function_seconds"],
                       "cost": run["container"]["estimated_compute_usd"], "puzzles": puzzles})
    return paired


def make_report(paths, output, benchmark=None, test=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    runs = [{"name": p.name, "answers": read_lines(p / "answers.jsonl"),
             "evaluations": read_lines(p / "evaluations.jsonl"),
             "summary": json.loads((p / "summary.json").read_text()),
             "metadata": json.loads((p / "metadata.json").read_text()),
             "container": json.loads((p / "container.json").read_text())} for p in paths]
    blue, gray = "#285cce", "#68758a"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 10,
        "text.color": "#253247", "axes.labelcolor": "#596579",
        "xtick.color": "#596579", "ytick.color": "#596579",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "axes.titleweight": "normal", "axes.titlelocation": "left",
        "axes.titlesize": 17, "axes.titlepad": 24,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none",
        "svg.hashsalt": "countdown", "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.major.pad": 10, "ytick.major.pad": 10,
        "axes.labelpad": 14, "axes.axisbelow": True,
    })
    devices = [next((g for g in ("H100", "L4", "A10") if g in r["metadata"]["gpu"]),
                    r["metadata"]["gpu"]) for r in runs]
    context = "Qwen3-0.6B on " + ", ".join(dict.fromkeys(devices))
    figures = []

    def save(fig, name):
        fig.savefig(output / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
        fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        figures.append(output / f"{name}.svg")

    for run in runs:
        if run["summary"]["mode"] != "calibrate":
            continue
        rows = run["answers"]
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["id"]].append(row["reward"])
        outcomes = Counter(int(sum(group)) for group in grouped.values())
        groups = len(grouped)
        mixed = sum(0 < sum(group) < len(group) for group in grouped.values())
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
        group_size = run["metadata"]["config"]["group_size"]
        xs = list(range(group_size + 1))
        axes[0].bar(xs, [outcomes[x] for x in xs],
                    color=["#dce2ea" if x in (0, group_size) else blue for x in xs])
        axes[0].set(xticks=xs, xlabel=f"Correct answers / {group_size}", ylabel="Prompts")
        axes[0].set_title(f"{mixed}/{groups} mixed-reward groups", fontsize=12)
        statuses = Counter(row["status"] for row in rows)
        names = list(statuses)
        axes[1].barh([s.replace("_", " ") for s in names], [statuses[s] for s in names],
                     color=[blue if s == "correct" else "#dce2ea" for s in names])
        axes[1].set(xlabel="Answers")
        axes[1].set_title(f"{statuses['correct']}/{len(rows)} correct", fontsize=12)
        axes[2].hist([r["tokens"] for r in rows], bins=np.arange(0, 69, 4), color=blue, rwidth=.85)
        axes[2].set(xlabel="Tokens, including EOS", ylabel="Answers")
        axes[2].set_title(f"{sum(not r['eos'] for r in rows)} hit the token cap", fontsize=12)
        fig.suptitle("Calibration", x=.04, ha="left", fontsize=17)
        save(fig, f"{run['name']}-calibration")

    training = sorted((r for r in runs if r["summary"]["mode"] == "train"),
                      key=lambda r: (r["metadata"]["config"]["seed"], "prompt_order" not in r["metadata"]))
    if training:
        seeds = sorted({r["metadata"]["config"]["seed"] for r in training})
        styles = {seed: ("-", "--", ":", "-.")[i % 4] for i, seed in enumerate(seeds)}
        fig, ax = plt.subplots(figsize=(10, 5.4))
        length_fig, length_ax = plt.subplots(figsize=(10, 4.3))
        highest = 0
        for run in training:
            points = [p for p in run["evaluations"] if p["split"] == "dev"]
            metadata = run["metadata"]
            rates = metadata["rates_usd_per_second"]
            rate = rates["gpu"] + metadata["cpu_cores"] * rates["cpu_core"] + metadata["ram_gib"] * rates["ram_gib"]
            cost = [p["elapsed_seconds"] * rate for p in points]
            accuracy = [100 * p["accuracy"] for p in points]
            highest = max(highest, max(accuracy))
            color = blue if "prompt_order" in metadata else gray
            style = styles[metadata["config"]["seed"]]
            ax.plot(cost, accuracy, color=color, ls=style, lw=2.2)
            ax.scatter(cost[-1], accuracy[-1], color=color, s=25, zorder=3)
            ax.annotate(f"{accuracy[-1]:.1f}%", (cost[-1], accuracy[-1]),
                        xytext=(7, 7), textcoords="offset points", color=color, fontsize=9)
            length_ax.plot([p["step"] for p in points], [p["mean_tokens"] for p in points],
                           color=color, ls=style, lw=2.2)
        legend = [Line2D([], [], color=color, lw=2.2, label=name)
                  for native, name, color in [(True, "grpo.cpp", blue), (False, "TRL", gray)]
                  if any(("prompt_order" in r["metadata"]) == native for r in training)]
        legend += [Line2D([], [], color="#a3aab5", lw=1.5, ls=style, label=f"Seed {seed}")
                   for seed, style in styles.items()]
        for chart in (ax, length_ax):
            chart.grid(axis="y", color="#e9edf2", lw=.7)
            chart.legend(handles=legend, loc="upper left", bbox_to_anchor=(0, 1.02),
                         ncol=4, frameon=False, fontsize=9, borderaxespad=0,
                         handlelength=2, columnspacing=2)
        ax.set(title="Learning Countdown", xlabel="Estimated compute (USD)",
               ylabel="Development accuracy (%)", xlim=(0, max(p.get_xdata()[-1] for p in ax.lines)*1.12),
               ylim=(0, min(100, max(50, 10*np.ceil(highest/10) + 10))))
        ax.xaxis.set_major_formatter(StrMethodFormatter("${x:.2f}"))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        fig.subplots_adjust(left=.09, right=.96, top=.80, bottom=.20)
        fig.text(.09, .025, context + ". Cost through each evaluation.", fontsize=9, color="#68758a")
        save(fig, "learning")

        fig.set_size_inches(4.8, 4.8)
        ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0, 1.13),
                  ncol=2, frameon=False, fontsize=9, borderaxespad=0)
        ax.set_title("Learning Countdown", fontsize=16, pad=48)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        fig.subplots_adjust(left=.15, right=.93, top=.72, bottom=.20)
        fig.texts[0].set_position((.15, .025))
        fig.texts[0].set_text(context + ".\nCost through each evaluation.")
        save(fig, "learning-mobile")

        length_ax.set(title="Completion length", xlabel="Optimizer updates",
                      ylabel="Tokens per answer", xticks=[0, 16, 32, 48, 64], xlim=(0, 64))
        length_ax.margins(y=.2)
        length_ax.yaxis.set_major_locator(MaxNLocator(4))
        length_fig.subplots_adjust(left=.09, right=.96, top=.77, bottom=.20)
        save(length_fig, "length")

        fig, ax = plt.subplots(figsize=(10, max(3.3, len(training)*.5 + 1.6)))
        stages = [("generation", "Generate", blue), ("scoring_forward", "Score", "#79a5d8"),
                  ("backward_and_batch_overhead", "Backward + prep", "#253d65"),
                  ("optimizer", "Adam", "#d5a34d"), ("evaluation", "Evaluate", "#9aa8bc")]
        left = np.zeros(len(training))
        positions = np.arange(len(training))
        for key, label, color in stages:
            values = np.array([r["summary"]["stage_seconds"].get(key, 0) for r in training])
            ax.barh(positions, values, left=left, height=.42, label=label, color=color)
            left += values
        totals = np.array([r["container"]["function_seconds"] for r in training])
        ax.barh(positions, np.maximum(0, totals-left), left=left, height=.42,
                label="Other", color="#dce2ea")
        names = [f"{'grpo.cpp' if 'prompt_order' in r['metadata'] else 'TRL'} {r['metadata']['config']['seed']}"
                 for r in training]
        for y, total in enumerate(totals):
            ax.text(total + max(totals)*.015, y, f"{total:.0f}s", va="center", fontsize=10)
        ax.set(yticks=positions, yticklabels=names, xlabel="Seconds", title="Training time on " + ", ".join(dict.fromkeys(devices)),
               xlim=(0, max(totals)*1.12))
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e9edf2", lw=.7)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.08), ncol=6,
                  frameon=False, fontsize=8, borderaxespad=0, handlelength=1.2, columnspacing=1.6)
        fig.subplots_adjust(left=.14, right=.96, top=.72, bottom=.20)
        save(fig, "runtime")

    if benchmark:
        samples = json.loads((benchmark / "samples.json").read_text())
        device = json.loads((benchmark / "metadata.json").read_text())["gpu"]
        fig, ax = plt.subplots(figsize=(10, 3.4))
        for y, name, color in [(1, "native", blue), (0, "trl", gray)]:
            values = [s for row in samples if row["implementation"] == name for s in row["seconds"]]
            median = np.median(values)
            ax.hlines(y, min(values), max(values), color=color, lw=1)
            ax.scatter(values, y + np.linspace(-.08, .08, len(values)), color=color, s=18, alpha=.65)
            ax.scatter(median, y, color=color, marker="|", s=250, linewidths=2)
            ax.text(max(values)+.05, y, f"{median:.3f}s", va="center", color=color, fontsize=12)
        ax.set(yticks=[0, 1], yticklabels=["TRL / BF16", "grpo.cpp / FP16"],
               xlabel="Forward + loss + backward (seconds)", xlim=(0, None), ylim=(-.45, 1.5),
               title=f"Same 32 completions on {device}")
        ax.margins(x=.1)
        ax.grid(axis="x", color="#e9edf2", lw=.7)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        fig.subplots_adjust(left=.17, right=.92, top=.80, bottom=.24)
        save(fig, "fixed-work")

    pairs = []
    if test:
        metrics = read_lines(test / "evaluations.jsonl")
        answers = read_lines(test / "answers.jsonl")
        pairs = pair_answers(metrics, answers, runs)
        example = next((p for p in pairs if p["name"].startswith("native")), pairs[0])
        categories = [("invalid", "Malformed / wrong numbers", "#dce2ea"),
                      ("wrong_value", "Wrong value", "#8b99ae"), ("correct", "Correct", blue)]
        fig, ax = plt.subplots(figsize=(10, 3.4))
        for y, stage in enumerate(("before", "after")):
            counts = Counter(p[stage]["status"] if p[stage]["status"] in ("wrong_value", "correct")
                             else "invalid" for p in example["puzzles"])
            left = 0
            for key, label, color in categories:
                value = counts[key]
                ax.barh(y, value, left=left, height=.42, color=color, label=label if y == 0 else None)
                if value:
                    ax.text(left + value/2, y + (.37 if value < 15 else 0), str(value),
                            ha="center", va="center", fontsize=10,
                            color="white" if key != "invalid" and value >= 15 else "#253247")
                left += value
        ax.set(yticks=[0, 1], yticklabels=["Before", f"After {example['step']} updates"],
               xlim=(0, len(example["puzzles"])), ylim=(1.65, -.7), xticks=[],
               title="What changed in the answers")
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.12), ncol=3, frameon=False,
                  fontsize=9, borderaxespad=0, handlelength=1.2, columnspacing=2)
        fig.subplots_adjust(left=.17, right=.96, top=.70, bottom=.12)
        fig.text(.17, .03, f"{len(example['puzzles'])} held-out puzzles. {example['label']}. {context}.",
                 fontsize=9, color=gray)
        save(fig, "outcomes")

    images, details = [], []
    for path in figures:
        if path.stem == "learning-mobile":
            continue
        tag = ('<img alt="' + html.escape(path.stem.replace("-", " ")) + '" src="data:image/svg+xml;base64,'
               + base64.b64encode(path.read_bytes()).decode() + '">')
        if path.stem == "learning":
            mobile = base64.b64encode(path.with_name("learning-mobile.svg").read_bytes()).decode()
            tag = '<picture><source media="(max-width: 600px)" srcset="data:image/svg+xml;base64,' + mobile + '">' + tag + '</picture>'
        tag = '<div class="plot">' + tag + '</div>'
        (details if path.stem in ("length", "runtime", "fixed-work") else images).append(tag)
    table = ""
    for run in runs:
        summary = run["summary"]
        if "sampled_device_used_bytes" in summary:
            memory = summary["sampled_device_used_bytes"] / 2**30
            memory_label = "sampled device"
        else:
            peak = summary.get("training_peak")
            memory = (peak["reserved_bytes"] if peak else summary["peak_reserved_bytes"]) / 2**30
            memory_label = "PyTorch reserved"
        table += (f"<tr><td>{html.escape(run['name'])}</td><td>{html.escape(run['metadata']['gpu'])}</td>"
                  f"<td>{html.escape(run['summary']['status'])}</td>"
                  f"<td>{run['container']['function_seconds']/60:.1f} min</td>"
                  f"<td>${run['container']['estimated_compute_usd']:.3f}</td>"
                  f"<td>{memory:.2f} GiB<br><small>{memory_label}</small></td></tr>")
    # Model output is untrusted text. Escape the script boundary and render with textContent.
    answer_fields = ("id", "nums", "target", "step", "text", "status", "value", "reward", "tokens", "eos")
    viewed = [{"name": r["name"], "answers": [{k: a[k] for k in answer_fields} for a in r["answers"]]}
              for r in runs]
    test_table = ""
    if test:
        test_table = "<h2>Final test</h2><p class=note>512 puzzles, one sampled answer each. Models saved at update 64.</p>"
        test_table += "<div class=table-scroll><table><tr><th>Run / seed</th><th>Before → after</th><th>Gains</th><th>Regressions</th></tr>"
        for row in metrics:
            if row["step"] == 0:
                continue
            before = next(r for r in metrics if r["step"] == 0 and r["seed"] == row["seed"]
                          and r["implementation"] == row["implementation"])
            test_table += (f"<tr><td>{'grpo.cpp' if row['implementation'] == 'native' else 'TRL'} {html.escape(str(row['seed']))}</td>"
                           f"<td>{before['statuses'].get('correct', 0)}/512 → {row['statuses'].get('correct', 0)}/512</td>"
                           f"<td>{row['gains']}</td><td>{row['regressions']}</td></tr>")
        test_table += "</table></div>"
    benchmark_note = ("<p class=note>The fixed-batch check times 15 batches per implementation across three alternating rounds "
                      "after warmup. Weights stay fixed; generation and Adam are excluded. TRL includes entropy, clipping "
                      "metrics and one extra backbone position; native includes host copies.</p>" if benchmark else "")
    page = Path(__file__).with_name("viewer.html").read_text()
    replacements = {"TABLE": table, "IMAGES": "".join(images), "DETAILS": "".join(details),
                    "CONTEXT": html.escape(context), "BENCHMARKNOTE": benchmark_note,
                    "FINALTEST": f"<section>{test_table}</section>" if test_table else "",
                    "PAYLOAD": json.dumps(viewed).replace("<", "\\u003c"),
                    "PAIRS": json.dumps(pairs).replace("<", "\\u003c")}
    page = re.sub(r"\{\{([A-Z]+)\}\}", lambda match: replacements[match[1]], page)
    (output / "report.html").write_text(page)
    return output / "report.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--test", type=Path)
    args = parser.parse_args()
    print(make_report(args.runs, args.output, args.benchmark, args.test))
