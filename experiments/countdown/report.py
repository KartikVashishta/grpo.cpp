"""Plot a Countdown run and browse its saved answers."""

import argparse
import base64
from collections import Counter, defaultdict
import html
import json
from pathlib import Path


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


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
        ax.set(title="Accuracy vs. cost", xlabel="Estimated compute (USD)",
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
        ax.set_title("Accuracy vs. cost", fontsize=16, pad=48)
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
    viewed = [{k: r[k] for k in ("name", "answers")} for r in runs]
    test_table = ""
    if test:
        metrics = read_lines(test / "evaluations.jsonl")
        answers = read_lines(test / "answers.jsonl")
        test_table = "<h2>Final test</h2><p class=note>512 puzzles, one sampled answer each. Models saved at update 64.</p>"
        test_table += "<div class=table-scroll><table><tr><th>Run / seed</th><th>Before → after</th><th>Gains</th><th>Regressions</th></tr>"
        for row in metrics:
            viewed.append({"name": row["run"] + " · test", "answers": [a for a in answers if a["run"] == row["run"]]})
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
    payload = json.dumps(viewed).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Countdown</title>
<style>
* { box-sizing: border-box; }
body { margin: 0 auto; padding: 56px 28px 64px; max-width: 1000px; color: #253247;
       background: white; font: 15px/1.6 "Helvetica Neue", Arial, sans-serif; }
header { margin-bottom: 28px; }
h1 { margin: 0; font-size: 40px; line-height: 1.15; font-weight: 600; letter-spacing: -1.4px; }
header p { margin: 10px 0 0; color: #68758a; }
h2 { margin: 0 0 8px; font-size: 23px; font-weight: 500; letter-spacing: -.4px; }
p { margin: 0 0 20px; }
img { display: block; width: 100%; height: auto; }
.plot { overflow-x: auto; }
section { margin-top: 48px; }
.note { color: #68758a; font-size: 13px; max-width: 76ch; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 14px 10px; text-align: right; border-bottom: 1px solid #e9edf2; white-space: nowrap; }
th { color: #68758a; font-size: 12px; font-weight: 500; }
th:first-child, td:first-child { text-align: left; padding-left: 0; }
.table-scroll { overflow-x: auto; }
details { margin-top: 32px; padding: 20px 0; border-block: 1px solid #e9edf2; }
summary { cursor: pointer; font-size: 14px; color: #596579; }
details[open] summary { margin-bottom: 28px; }
details img { margin: 24px 0; }
details .note { margin-top: 24px; }
.controls { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 24px; }
.controls > div { display: grid; gap: 6px; }
label { font-size: 13px; color: #596579; }
select { width: 100%; min-width: 0; padding: 11px 12px; border: 1px solid #dce2ea;
         border-radius: 6px; background: white; color: #253247; font: inherit; font-size: 15px; }
.controls > .update { grid-column: 1 / -1; grid-template-columns: auto 1fr 28px; align-items: center; gap: 16px; }
input { width: 100%; margin: 0; accent-color: #285cce; }
:focus-visible { outline: 2px solid #285cce; outline-offset: 3px; }
.answer { margin-top: 28px; }
.answer strong { font-size: 13px; font-weight: 500; }
.correct { color: #237956; } .failure { color: #9b4c36; }
pre { margin: 10px 0 0; padding: 22px 24px; border-radius: 6px; background: #f5f7fb;
      white-space: pre-wrap; overflow-wrap: anywhere; font: 24px/1.5 "SFMono-Regular", Consolas, monospace; }
small { color: #68758a; }
@media (max-width: 600px) {
    body { padding: 28px 18px; }
    h1 { font-size: 32px; }
    .controls { grid-template-columns: 1fr; gap: 14px; }
    th, td { padding: 12px 8px; font-size: 12px; }
    pre { font-size: 20px; padding: 16px; }
    img { min-width: 640px; }
    picture img { min-width: 0; }
}
</style>
<header><h1>Countdown</h1><p>CONTEXT</p></header>
IMAGES
FINALTEST
<details><summary>Training details</summary>
DETAILS
<div class="table-scroll"><table>
<thead><tr><th>Run</th><th>GPU</th><th>Status</th><th>Time</th><th>Compute</th><th>Memory</th></tr></thead>
<tbody>TABLE</tbody></table></div>
<p class="note">Costs use recorded GPU, CPU and RAM rates. Curves stop at each evaluation;
the table includes loading, saving, reload checks and writing results. Builds, startup,
storage and network are excluded.</p>
<p class="note">Native memory is sampled device usage; TRL is peak reserved memory.
Both are measured before reload. Native uses FP16 compute; TRL uses BF16 with FP32 parameters
and its tiled head. Sampling RNGs differ. Two seeds do not establish equal learning quality.</p>
<p class="note">Calibration records eight answers per puzzle; development and test record one.
Token counts include EOS.</p>
BENCHMARKNOTE
</details>
<section><h2>Saved answers</h2>
<div class="controls">
<div><label for="run">Run</label><select id="run"></select></div>
<div><label for="puzzle">Puzzle</label><select id="puzzle"></select></div>
<div class="update"><label for="step">Update</label><input id="step" type="range" min="0" value="0"><output id="step-label" for="step"></output></div>
</div><div id="answers"></div>
</section>
<script>const runs=PAYLOAD;
const run = document.getElementById('run');
const puzzle = document.getElementById('puzzle');
const step = document.getElementById('step');
for (const [i, r] of runs.entries()) run.add(new Option(r.name, i));

function selectRun() {
    puzzle.replaceChildren();
    const unique = new Map(runs[run.value].answers.map(a => [a.id, a]));
    for (const a of unique.values()) {
        puzzle.add(new Option(a.nums.join(', ') + ' → ' + a.target, a.id));
    }
    selectPuzzle();
}

function selectPuzzle() {
    const rows = runs[run.value].answers.filter(a => a.id === puzzle.value);
    const steps = [...new Set(rows.map(a => a.step))].sort((a, b) => a - b);
    step.max = steps.length - 1;
    step.value = 0;
    render();
}

function render() {
    const rows = runs[run.value].answers.filter(a => a.id === puzzle.value);
    const steps = [...new Set(rows.map(a => a.step))].sort((a, b) => a - b);
    const selected = steps[step.value];
    step.setAttribute('aria-valuetext', String(selected));
    document.getElementById('step-label').textContent = ' ' + selected;
    const box = document.getElementById('answers');
    box.replaceChildren();
    for (const a of rows.filter(a => a.step === selected)) {
        const item = document.createElement('div');
        item.className = 'answer';
        const status = document.createElement('strong');
        status.className = a.reward ? 'correct' : 'failure';
        status.textContent = a.status.replaceAll('_', ' ') + ' · value ' + (a.value ?? '—')
            + ' · ' + a.tokens + ' tokens · ' + (a.eos ? 'EOS' : 'cap');
        const expression = document.createElement('pre');
        expression.textContent = a.text;
        item.append(status, expression);
        box.append(item);
    }
}

run.onchange = selectRun;
puzzle.onchange = selectPuzzle;
step.oninput = render;
selectRun();</script></html>'''
    (output / "report.html").write_text(page.replace("TABLE", table).replace("IMAGES", "".join(images))
                                       .replace("DETAILS", "".join(details)).replace("CONTEXT", html.escape(context))
                                       .replace("BENCHMARKNOTE", benchmark_note)
                                       .replace("FINALTEST", f"<section>{test_table}</section>" if test_table else "")
                                       .replace("PAYLOAD", payload))
    return output / "report.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--test", type=Path)
    args = parser.parse_args()
    print(make_report(args.runs, args.output, args.benchmark, args.test))
