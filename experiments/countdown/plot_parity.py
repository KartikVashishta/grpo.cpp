"""Plot saved native gradient checks; no model or GPU needed."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot(before, after, output):
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "countdown",
                         "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 10,
                         "text.color": "#253247", "axes.labelcolor": "#596579",
                         "xtick.color": "#596579", "ytick.color": "#596579",
                         "axes.titleweight": "normal", "axes.titlelocation": "left",
                         "axes.titlepad": 18, "xtick.major.size": 0, "ytick.major.size": 0})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.grid(axis="y", color="#e9edf2", linewidth=.7)
        ax.set_axisbelow(True)
    for name, label, color in [("unscaled", "No loss scaling", "#68758a"),
                                ("scaled", "Loss scale 256", "#285cce")]:
        tensors = json.loads((after / (name + ".json")).read_text())["tensors"]
        errors = [100 * max(t["relative_l2"] for t in tensors
                           if t["name"].startswith(f"model.layers.{layer}.")) for layer in range(28)]
        axes[0].plot(range(28), errors, label=label, color=color, linewidth=2)
    axes[0].set(title="Gradient error by layer", xlabel="Layer", ylabel="Max. relative L2 error (%)")
    axes[0].set_xlim(0, 27)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False)

    errors = [100 * json.loads((path / "summary.json").read_text())["cases"]["repeat"]["relative_l2"]
              for path in (before, after)]
    axes[1].bar(["FP16 atomics", "FP32 row sums"], errors, color=["#68758a", "#285cce"], width=0.55)
    axes[1].set_yscale("log")
    axes[1].set(title="Identical input, repeated", ylabel="Relative L2 difference (%)")
    axes[1].axhline(0.001, color="#777777", linestyle="--", linewidth=1)
    axes[1].text(1.45, 0.00125, "Fixed tolerance", ha="right", color="#666666", fontsize=9)
    for i, value in enumerate(errors):
        axes[1].annotate(f"{value:.2g}%", (i, value), xytext=(0, 5), textcoords="offset points", ha="center")
    axes[1].set_ylim(min(errors) / 3, max(errors) * 4)
    fig.suptitle("Qwen3-0.6B gradient check", x=0.07, ha="left", fontsize=17)
    fig.text(0.07, 0.02, "310 tensors against FP32 PyTorch on H100.", color="#555555", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.07, 1, 0.95), w_pad=2.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".png"):
        fig.savefig(output.with_suffix(suffix), dpi=180, facecolor=fig.get_facecolor(),
                    metadata={"Date": None} if suffix == ".svg" else None)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.before, args.after, args.output)
