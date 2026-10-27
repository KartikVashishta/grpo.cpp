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
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    runs = [{"name": p.name, "answers": read_lines(p / "answers.jsonl"),
             "evaluations": read_lines(p / "evaluations.jsonl"),
             "summary": json.loads((p / "summary.json").read_text()),
             "metadata": json.loads((p / "metadata.json").read_text()),
             "container": json.loads((p / "container.json").read_text())} for p in paths]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "#faf9f6",
                         "axes.facecolor": "#faf9f6", "axes.edgecolor": "#b6b5b1",
                         "svg.fonttype": "none", "svg.hashsalt": "countdown",
                         "savefig.facecolor": "#faf9f6"})
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
                    color=["#b8b6b0" if x in (0, group_size) else "#296b78" for x in xs])
        axes[0].set(xticks=xs, xlabel="Correct answers out of eight", ylabel="Prompts",
                    title=f"{mixed}/{groups} groups have learning signal")
        statuses = Counter(row["status"] for row in rows)
        names = list(statuses)
        axes[1].barh([s.replace("_", " ") for s in names], [statuses[s] for s in names],
                     color=["#296b78" if s == "correct" else "#b8b6b0" for s in names])
        axes[1].set(xlabel="Completions", title=f"{statuses['correct']}/{len(rows)} answers solve the task")
        axes[2].hist([r["tokens"] for r in rows], bins=np.arange(0, 69, 4), color="#296b78", rwidth=.85)
        axes[2].set(xlabel="Generated tokens, including EOS", ylabel="Completions",
                    title=f"{sum(not r['eos'] for r in rows)} truncated answers")
        fig.suptitle("Qwen3-0.6B · Countdown calibration · 128 prompts × 8 responses", fontsize=13)
        save(fig, f"{run['name']}-calibration")

    training = [r for r in runs if r["summary"]["mode"] == "train"]
    if training:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        upper = 0
        for index, run in enumerate(training):
            points = [p for p in run["evaluations"] if p["split"] == "dev"]
            rate = sum([run["metadata"]["rates_usd_per_second"]["gpu"],
                        run["metadata"]["cpu_cores"] * run["metadata"]["rates_usd_per_second"]["cpu_core"],
                        run["metadata"]["ram_gib"] * run["metadata"]["rates_usd_per_second"]["ram_gib"]])
            x = np.array([p["elapsed_seconds"] * rate for p in points])
            y = np.array([p["accuracy"] for p in points])
            n = np.array([p["prompts"] for p in points])
            # Wilson intervals describe finite held-out samples, not variation across training seeds.
            z = 1.96
            center = (y + z*z/(2*n)) / (1 + z*z/n)
            half = z*np.sqrt(y*(1-y)/n + z*z/(4*n*n)) / (1+z*z/n)
            upper = max(upper, max((center+half)*100))
            native = "prompt_order" in run["metadata"]
            seed = run["metadata"]["config"]["seed"]
            gpu = next((g for g in ("H100", "L4", "A10") if g in run["metadata"]["gpu"]), run["metadata"]["gpu"])
            label = f"{'Native' if native else 'TRL'} · {gpu} · seed {seed}"
            color = "#296b78" if native else "#b05b3c"
            style = ("o" if gpu == "H100" else "s") + ("-" if seed == 42 else "--")
            axes[0].plot(x, y*100, style, color=color, label=label, lw=2, ms=4)
            axes[0].fill_between(x, (center-half)*100, (center+half)*100, color=color, alpha=.1)
            axes[0].annotate(f"{round(y[-1]*n[-1])}/{n[-1]}", (x[-1], y[-1]*100),
                             xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
            axes[1].plot([p["step"] for p in points], [p["mean_tokens"] for p in points],
                         style, color=color, label=label, lw=2, ms=4)
        axes[0].set(xlabel="Estimated compute cost (USD)", ylabel="Development pass@1 (%)",
                    title="Accuracy against cost", ylim=(0, min(100, 10*np.ceil(upper/10))))
        axes[1].set(xlabel="Optimizer updates", ylabel="Mean generated tokens, including EOS",
                    title="Completion length")
        for ax in axes:
            ax.grid(axis="y", alpha=.15)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle("Countdown · sampled development answers", fontsize=14)
        fig.supxlabel("Evaluated every 16 updates. Bands: 95% Wilson intervals over prompts, not training-seed uncertainty.\n"
                      "Cost stops at each evaluation; excludes the final save, reload, artifact commit, image build and startup.", fontsize=8)
        save(fig, "learning")

        fig, ax = plt.subplots(figsize=(10, max(3, len(training) * .8 + 1.7)), layout="constrained")
        stages = [("generation", "Generation", "#296b78"), ("scoring_forward", "Scoring", "#7299a1"),
                  ("backward_and_batch_overhead", "Backward + batch overhead", "#b05b3c"),
                  ("optimizer", "Optimizer", "#d4a28b"), ("evaluation", "Evaluation", "#6a5892")]
        left = np.zeros(len(training))
        for key, label, color in stages:
            values = np.array([r["summary"]["stage_seconds"].get(key, 0)/60 for r in training])
            ax.barh([r["name"] for r in training], values, left=left, label=label, color=color)
            left += values
        rest = [max(0, r["container"]["function_seconds"]/60-left[i]) for i, r in enumerate(training)]
        ax.barh([r["name"] for r in training], rest, left=left, label="Load, save + other", color="#b8b6b0")
        ax.set(xlabel="Minutes", title="Where the runs spend their time")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=3, frameon=False, fontsize=9)
        save(fig, "runtime")

    if benchmark:
        samples = json.loads((benchmark / "samples.json").read_text())
        device = json.loads((benchmark / "metadata.json").read_text())["gpu"]
        fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
        for y, name, color in [(1, "native", "#296b78"), (0, "trl", "#b05b3c")]:
            values = [s for row in samples if row["implementation"] == name for s in row["seconds"]]
            median = np.median(values)
            ax.barh(y, median, height=.32, color=color, alpha=.2)
            ax.scatter(values, y + np.linspace(-.09, .09, len(values)), color=color, s=22)
            ax.annotate(f"{median:.3f} s", (median, y), xytext=(8, 18), textcoords="offset points", color=color)
        ax.set(yticks=[0, 1], yticklabels=["TRL · BF16 compute", "Native · FP16 compute"],
               xlabel="Seconds per forward + loss + backward batch", xlim=(0, None), ylim=(-.45, 1.5),
               title=f"Same 32 recorded completions · {device}")
        ax.grid(axis="x", alpha=.15)
        fig.supxlabel("15 samples each, after warmup. Three rounds with alternating order; weights stay fixed.\n"
                      "TRL includes its tiled head, entropy and one extra backbone position. No optimizer step is timed.", fontsize=8)
        save(fig, "fixed-work")

    image_tags = "".join('<img alt="Measured Countdown results" src="data:image/svg+xml;base64,' +
                         base64.b64encode(p.read_bytes()).decode() + '">' for p in figures)
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
        test_table = "<h2>Final test</h2><p>512 frozen puzzles, one sampled answer per puzzle. Every trained checkpoint is from update 64.</p>"
        test_table += "<table><tr><th>Run</th><th>Before → after</th><th>Gains</th><th>Regressions</th></tr>"
        for row in metrics:
            viewed.append({"name": row["run"] + " · test", "answers": [a for a in answers if a["run"] == row["run"]]})
            if row["step"] == 0:
                continue
            before = next(r for r in metrics if r["step"] == 0 and r["seed"] == row["seed"]
                          and r["implementation"] == row["implementation"])
            test_table += (f"<tr><td>{html.escape(row['run'])}</td>"
                           f"<td>{before['statuses'].get('correct', 0)}/512 → {row['statuses'].get('correct', 0)}/512</td>"
                           f"<td>{row['gains']}</td><td>{row['regressions']}</td></tr>")
        test_table += "</table>"
    payload = json.dumps(viewed).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Countdown</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;background:#faf9f6;color:#202b30;max-width:1100px;margin:48px auto;padding:0 24px}
h1{font-size:38px;letter-spacing:-1px;margin-bottom:8px}h2{font-size:24px;margin-top:40px}p{max-width:820px}
img{width:100%;height:auto;margin:18px 0}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #d8d7d3;padding:12px 10px}
label{display:block;margin:14px 0}select{max-width:100%;font:inherit;padding:8px;background:white;border:1px solid #aaa;border-radius:4px}
input{width:100%;max-width:520px;vertical-align:middle}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:18px/1.6 monospace;background:#efeee9;padding:18px}
.answer{border-top:1px solid #d8d7d3;padding:14px 0}.correct{color:#17616d}.failure{color:#9a452b}.note{font-size:13px;color:#566067}
</style><h1>Countdown</h1>
<p>Qwen3-0.6B, one GPU, exact arithmetic rewards.</p>
<table><thead><tr><th>Run</th><th>GPU</th><th>Status</th><th>Function time</th><th>Estimated compute</th><th>Memory</th></tr></thead>
<tbody>TABLE</tbody></table>
<p class="note">Cost is a rate-based estimate. TRL records peak PyTorch reserved memory;
the native run samples whole-device usage at stage boundaries. These measure different things.
Both exclude checkpoint-reload verification.</p>
<p class="note">Each curve is one training run. Native and TRL use different sampling RNGs and
compute precision. These runs do not establish equal learning quality.
Curve costs stop at each evaluation; the table includes the full function, including saves,
reload and the first artifact commit.</p>
IMAGES
FINALTEST
<h2>Answers</h2>
<label>Run <select id="run"></select></label><label>Puzzle <select id="puzzle"></select></label>
<label>Recorded update <input id="step" type="range" min="0" value="0"><span id="step-label"></span></label>
<div id="answers"></div><p class="note">Calibration has eight samples per puzzle. Development and final testing have one.
Token counts include EOS. Only recorded checkpoints appear here; no GPU or network call is made.</p>
<script>const runs=PAYLOAD;
const run=document.getElementById('run'),puzzle=document.getElementById('puzzle'),step=document.getElementById('step');
for(const [i,r] of runs.entries())run.add(new Option(r.name,i));
function selectRun(){puzzle.replaceChildren();const unique=new Map(runs[run.value].answers.map(a=>[a.id,a]));
for(const a of unique.values())puzzle.add(new Option(a.nums.join(', ')+' → '+a.target,a.id));selectPuzzle();}
function selectPuzzle(){const rows=runs[run.value].answers.filter(a=>a.id===puzzle.value);
const steps=[...new Set(rows.map(a=>a.step))].sort((a,b)=>a-b);step.max=steps.length-1;step.value=0;render();}
function render(){const rows=runs[run.value].answers.filter(a=>a.id===puzzle.value);
const steps=[...new Set(rows.map(a=>a.step))].sort((a,b)=>a-b),selected=steps[step.value];
document.getElementById('step-label').textContent=' '+selected;const box=document.getElementById('answers');box.replaceChildren();
for(const a of rows.filter(a=>a.step===selected)){const item=document.createElement('div');item.className='answer';
const status=document.createElement('strong');status.className=a.reward?'correct':'failure';
status.textContent=a.status.replaceAll('_',' ')+' · value '+(a.value??'—')+' · '+a.tokens+' tokens · '+(a.eos?'EOS':'cap');
const expression=document.createElement('pre');expression.textContent=a.text;item.append(status,expression);box.append(item);}}
run.onchange=selectRun;puzzle.onchange=selectPuzzle;step.oninput=render;selectRun();</script></html>'''
    (output / "report.html").write_text(page.replace("TABLE", table).replace("IMAGES", image_tags)
                                       .replace("FINALTEST", test_table).replace("PAYLOAD", payload))
    return output / "report.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--test", type=Path)
    args = parser.parse_args()
    print(make_report(args.runs, args.output, args.benchmark, args.test))
