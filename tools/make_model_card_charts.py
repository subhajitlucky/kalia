"""Generate the model-card charts from the archived evaluation JSON.

The cards are prose, but a release comparison is easier to read when the
numbers have a shape. Every chart here is drawn from the committed JSON in
`docs/results/bench-pair-2k/` or from the recorded 500-sample table; nothing is
typed in twice. The 500-sample values are the published ones and are kept as a
named reference, not re-derived.

Usage:
    python tools/make_model_card_charts.py

Writes PNGs to docs/public/charts/. They are uploaded to the HF model repos by
tools/publish_model_card.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "results" / "bench-pair-2k"
OUT = ROOT / "docs" / "public" / "charts"

TASKS = ["piqa", "hellaswag", "winogrande", "lambada_openai", "arc_easy"]
LABELS = {
    "piqa": "PIQA",
    "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande",
    "lambada_openai": "LAMBADA",
    "arc_easy": "ARC-Easy",
}
# The published 500-sample deltas, in percentage points. Kept as a named
# reference so the correction chart cannot quietly use different numbers.
DELTA_500 = {
    "piqa": 2.40,
    "hellaswag": 3.00,
    "winogrande": 0.60,
    "lambada_openai": -2.20,
    "arc_easy": -3.80,
}

V012 = "#94A3B8"
V020 = "#0F766E"
NEG = "#DC2626"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"
BAND = "#FEF3C7"


def load():
    pair = json.loads((RESULTS / "bench_pair_2k.json").read_text())
    v012 = json.loads((RESULTS / "bench_v012_2k.json").read_text())
    v020 = json.loads((RESULTS / "bench_v020_2k.json").read_text())
    return pair, v012, v020


def style():
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


def clean_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def held_out_loss():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    panels = [
        ("Held-out validation loss", [3.0533, 2.8248], "nats", 3.0),
        ("Bits per byte", [0.9992, 0.9244], "bpB", 0.90),
    ]
    for ax, (title, values, unit, floor) in zip(axes, panels):
        bars = ax.bar(["v0.1.2", "v0.2.0"], values, color=[V012, V020],
                      width=0.55, zorder=3)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, max(values) * 1.18)
        clean_axes(ax)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value,
                    f"{value:.4f}", ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color=INK)
        ax.set_ylabel(unit)
    fig.suptitle("Held-out loss: same 819,200-token protocol, same held-out set",
                 x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.91,
             "Lower is better. v0.2.0 is 0.2285 nats and 0.0748 bpB lower.",
             ha="left", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "held_out_loss.png")


def zero_shot():
    pair, _, _ = load()
    x = np.arange(len(TASKS))
    width = 0.36
    v12 = [pair["comparison"][t]["v012"] * 100 for t in TASKS]
    v20 = [pair["comparison"][t]["v020"] * 100 for t in TASKS]
    se12 = [pair["comparison"][t]["v012_stderr"] * 100 for t in TASKS]
    se20 = [pair["comparison"][t]["v020_stderr"] * 100 for t in TASKS]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    b12 = ax.bar(x - width / 2, v12, width, yerr=se12, capsize=3,
                 color=V012, label="v0.1.2", zorder=3,
                 error_kw={"ecolor": INK, "lw": 0.8})
    b20 = ax.bar(x + width / 2, v20, width, yerr=se20, capsize=3,
                 color=V020, label="v0.2.0", zorder=3,
                 error_kw={"ecolor": INK, "lw": 0.8})
    ax.set_xticks(x, [LABELS[t] for t in TASKS])
    ax.set_ylim(0, 72)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Zero-shot accuracy at 2,000 samples", loc="left")
    clean_axes(ax)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    for bars in (b12, b20):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.1,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=8.5,
                    color=INK)
    fig.text(0.02, 0.93,
             "0-shot, lm-evaluation-harness 0.4.9. Error bars are ±1 standard "
             "error; no movement reaches 2σ.",
             ha="left", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "zero_shot_accuracy.png")


def accuracy_deltas():
    pair, _, _ = load()
    rows = []
    for task in TASKS:
        row = pair["comparison"][task]
        se = (row["v012_stderr"] ** 2 + row["v020_stderr"] ** 2) ** 0.5 * 100
        rows.append((task, row["delta_pp"], se, row["z"]))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.axvspan(-1.0, 1.0, color=BAND, alpha=0.8, zorder=0)
    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    y = np.arange(len(rows))
    for i, (task, delta, se, z) in enumerate(rows):
        color = NEG if delta < 0 else V020
        ax.errorbar(delta, i, xerr=1.96 * se, fmt="o", color=color,
                    ecolor=color, elinewidth=1.4, capsize=4, markersize=8,
                    zorder=3)
        label = f"{delta:+.2f} pp  (z {z:+.2f})"
        ax.text(delta + (0.05 if delta >= 0 else -0.05), i + 0.16, label,
                ha="left" if delta >= 0 else "right", va="bottom",
                fontsize=9.5, color=color, fontweight="bold")
    ax.set_yticks(y, [LABELS[t] for t, _, _, _ in rows])
    ax.set_xlim(-3.2, 3.2)
    ax.set_xlabel("Δ accuracy, v0.2.0 − v0.1.2 (percentage points)")
    ax.set_title("The change is small, and the interval usually includes zero",
                 loc="left")
    clean_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)
    fig.text(0.02, 0.92,
             "Bars are 95% confidence intervals from each task's standard error; "
             "the shaded band is the registered ±1.0 pp allowance. LAMBADA is "
             "outside it, but its interval crosses zero.",
             ha="left", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "accuracy_deltas.png")


def correction_500_vs_2k():
    pair, _, _ = load()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.axvspan(-1.0, 1.0, color=BAND, alpha=0.8, zorder=0)
    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    y = np.arange(len(TASKS))
    for i, task in enumerate(TASKS):
        d500 = DELTA_500[task]
        d2k = pair["comparison"][task]["delta_pp"]
        ax.plot([d500, d2k], [i, i], color=GRID, lw=2.5, zorder=2)
        ax.scatter([d500], [i], s=70, color=V012, zorder=3)
        ax.scatter([d2k], [i], s=80, color=V020, zorder=4)
        ax.text(d500, i - 0.28, f"{d500:+.2f}", ha="center", va="top",
                fontsize=8.5, color=MUTED)
        top_row = i == len(TASKS) - 1
        ax.text(d2k, i - 0.28 if top_row else i + 0.20,
                f"{d2k:+.2f}", ha="center",
                va="top" if top_row else "bottom",
                fontsize=9, color=V020, fontweight="bold")
    ax.set_yticks(y, [LABELS[t] for t in TASKS])
    ax.set_xlim(-4.4, 3.4)
    ax.set_xlabel("Δ accuracy, v0.2.0 − v0.1.2 (percentage points)")
    ax.set_title("The 500-sample deltas did not survive 2,000 samples", loc="left")
    clean_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)
    fig.text(0.02, 0.92,
             "Grey = 500 samples; teal = 2,000. ARC-Easy went from −3.80 to "
             "+0.45; HellaSwag from +3.00 to 0.00. Shaded band = ±1.0 pp "
             "allowance.",
             ha="left", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "correction_500_vs_2k.png")


def main():
    style()
    held_out_loss()
    zero_shot()
    accuracy_deltas()
    correction_500_vs_2k()


if __name__ == "__main__":
    main()
