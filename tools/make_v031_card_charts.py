"""Generate the v0.3.1 model-card charts from archived JSON.

Three static charts, same visual language as the v0.2.0 card:

1. held-out loss / bpB across v0.1.2, v0.2.0, v0.3.0, v0.3.1;
2. 2,000-sample accuracy deltas vs v0.2.0 and vs v0.3.0;
3. CL-0 probe loss for v0.2.0, v0.3.0, v0.3.1.

Nothing is typed in twice: the 2k numbers come from the archived JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "public" / "charts"

V012 = "#94A3B8"
V020 = "#0F766E"
V030 = "#2563EB"
V031 = "#7C3AED"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

TASKS = ["arc_easy", "hellaswag", "lambada_openai", "piqa", "winogrande"]
LABELS = {
    "arc_easy": "ARC-Easy",
    "hellaswag": "HellaSwag",
    "lambada_openai": "LAMBADA",
    "piqa": "PIQA",
    "winogrande": "WinoGrande",
}
METRIC = {
    "arc_easy": "acc,none",
    "hellaswag": "acc_norm,none",
    "lambada_openai": "acc,none",
    "piqa": "acc,none",
    "winogrande": "acc,none",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def load_2k() -> tuple[dict, dict, dict]:
    v020 = json.loads((ROOT / "docs/results/bench-pair-2k/bench_v020_2k.json").read_text())
    v030 = json.loads((ROOT / "docs/results/v030-epoch2/final/bench_v030_2k.json").read_text())
    v031 = json.loads((ROOT / "docs/results/v031-merge/bench_v031_2k.json").read_text())
    return v020, v030, v031


def held_out() -> None:
    models = ["v0.1.2", "v0.2.0", "v0.3.0", "v0.3.1"]
    val = [3.0533, 2.8248, 2.7743, 2.7579]
    bpb = [0.9992, 0.9244, 0.9070, 0.9016]
    colors = [V012, V020, V030, V031]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, values, unit, floor, title in (
        (axes[0], val, "nats", 2.70, "Held-out validation loss"),
        (axes[1], bpb, "bpB", 0.89, "Bits per byte"),
    ):
        x = np.arange(len(models))
        bars = ax.bar(x, values, color=colors, width=0.58, zorder=3)
        ax.set_xticks(x, models)
        ax.set_ylim(floor, max(values) * 1.015)
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
        clean(ax)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value,
                    f"{value:.4f}", ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold", color=INK)
    fig.suptitle("Held-out loss: 819,200 tokens, seed 1234 — lower is better",
                 x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.91,
             "v0.3.1 is −0.0669 nats vs v0.2.0 and −0.0164 vs v0.3.0. "
             "The y-axis starts at 2.70 / 0.89 to make the small differences visible.",
             ha="left", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save(fig, "v031_held_out_loss.png")


def zero_shot_deltas() -> None:
    v020, v030, v031 = load_2k()
    labels = [LABELS[t] for t in TASKS]
    d020 = [(v031[t][METRIC[t]] - v020[t][METRIC[t]]) * 100 for t in TASKS]
    d030 = [(v031[t][METRIC[t]] - v030[t][METRIC[t]]) * 100 for t in TASKS]
    y = np.arange(len(TASKS))
    height = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    b1 = ax.barh(y - height / 2, d020, height, color=V020,
                 label="vs v0.2.0 (last release)", zorder=3)
    b2 = ax.barh(y + height / 2, d030, height, color=V031,
                 label="vs v0.3.0 (merge base)", zorder=3)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy delta (percentage points)")
    ax.set_title("2,000-sample accuracy deltas — positive is better", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for bars in (b1, b2):
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{width:+.2f}", va="center", fontsize=8.8, color=INK)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, max(d020 + d030) * 1.18)
    fig.text(0.02, 0.93,
             "0-shot, lm-evaluation-harness 0.4.9, 2,000 samples. The v0.3.0 "
             "deltas isolate the merge; none reaches 2σ.",
             ha="left", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "v031_zero_shot_deltas.png")


def cl0() -> None:
    models = ["v0.2.0", "v0.3.0", "v0.3.1"]
    values = [2.798875, 2.756648, 2.740949]
    colors = [V020, V030, V031]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    bars = ax.bar(models, values, color=colors, width=0.55, zorder=3)
    ax.set_ylim(2.70, 2.81)
    ax.set_ylabel("probe loss (lower is better)")
    ax.set_title("CL-0 retention probe", loc="left")
    clean(ax)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value,
                f"{value:.6f}", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color=INK)
    fig.text(0.02, 0.90,
             "25 batches × 8 × 1,024, seed 1234. v0.3.1 improves by 0.0157 vs v0.3.0.",
             ha="left", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "v031_cl0.png")


def main() -> None:
    style()
    held_out()
    zero_shot_deltas()
    cl0()


if __name__ == "__main__":
    main()
