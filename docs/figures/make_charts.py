"""Generate the portfolio figures as dependency-free SVG.

Reads the real training logs (CSV) for the loss curve; the ablation and
benchmark values are transcribed from the journal entries. Run:

    python docs/figures/make_charts.py --logs /path/to/logs --out public/kalia
"""

import argparse
import csv
from pathlib import Path

TEXT = "#111827"
MUTED = "#6b7280"
LINE = "#e5e7eb"
ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
FONT = "system-ui,-apple-system,'Segoe UI',Roboto,sans-serif"


def esc(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, value, size=12, fill=TEXT, anchor="start", weight="400"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(value)}</text>'
    )


def rect(x, y, w, h, fill, opacity=1.0, rx=0):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" opacity="{opacity}" rx="{rx}"/>'
    )


def line(x1, y1, x2, y2, stroke=LINE, width=1, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{extra}/>'
    )


def svg_open(width, height):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
    )


def val_loss_chart(points, deterministic, out: Path):
    width, height = 720, 400
    left, right, top, bottom = 64, 28, 56, 56
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = 1600, 3600
    y_min, y_max = 2.35, 2.70

    def sx(step):
        return left + (step - x_min) / (x_max - x_min) * plot_w

    def sy(loss):
        return top + (y_max - loss) / (y_max - y_min) * plot_h

    parts = [svg_open(width, height)]
    parts.append(text(left, 24, "Validation loss by step", 15, TEXT, weight="600"))
    parts.append(text(left, 42, "training evals, 50 batches each", 12, MUTED))

    parts.append(rect(sx(1750), sy(2.6334), sx(2500) - sx(1750), sy(2.5270) - sy(2.6334), MUTED, 0.10))
    parts.append(text(sx(2125), sy(2.645) + 14, "plateau band 2.53 - 2.63", 11, MUTED, "middle"))

    for value in (2.4, 2.5, 2.6, 2.7):
        parts.append(line(left, sy(value), width - right, sy(value)))
        parts.append(text(left - 8, sy(value) + 4, f"{value:.1f}", 11, MUTED, "end"))
    for step in (2000, 2500, 3000, 3500):
        parts.append(line(sx(step), top, sx(step), height - bottom))
        parts.append(text(sx(step), height - bottom + 18, str(step), 11, MUTED, "middle"))
    parts.append(text(left + plot_w / 2, height - 16, "step", 12, MUTED, "middle"))

    path = " ".join(
        ("M" if i == 0 else "L") + f"{sx(step):.1f},{sy(loss):.1f}"
        for i, (step, loss) in enumerate(points)
    )
    parts.append(f'<path d="{path}" fill="none" stroke="{ACCENT}" stroke-width="2"/>')
    for step, loss in points:
        parts.append(f'<circle cx="{sx(step):.1f}" cy="{sy(loss):.1f}" r="4" fill="{ACCENT}"/>')

    dstep, dloss = deterministic
    parts.append(line(sx(dstep), top, sx(dstep), height - bottom, ACCENT_DARK, 1, "4 4"))
    parts.append(
        f'<rect x="{sx(dstep) - 5:.1f}" y="{sy(dloss) - 5:.1f}" width="10" height="10" '
        f'fill="{ACCENT_DARK}"/>'
    )
    parts.append(text(sx(dstep) - 8, sy(dloss) - 12, f"deterministic eval: {dloss}", 11, ACCENT_DARK, "end"))
    parts.append(text(sx(dstep) - 8, sy(dloss) + 20, "quota stop, step 3,478", 11, MUTED, "end"))
    parts.append("</svg>")
    out.write_text("".join(parts))


def bar_chart(title, subtitle, rows, out: Path, y_floor, y_step, chance=None, note=None):
    width = 720
    row_h = 44
    left, right, top = 210, 96, 64
    height = top + row_h * len(rows) + 46
    plot_w = width - left - right

    def bar_x(value):
        return left + (value - y_floor) / (y_step - y_floor) * plot_w

    parts = [svg_open(width, height)]
    parts.append(text(left, 26, title, 15, TEXT, weight="600"))
    parts.append(text(left, 44, subtitle, 12, MUTED))

    for i, (label, value, color) in enumerate(rows):
        y = top + i * row_h
        parts.append(text(left - 12, y + 22, label, 12, TEXT, "end"))
        parts.append(rect(left, y + 6, bar_x(value) - left, 24, color, 0.9, 3))
        parts.append(text(bar_x(value) + 8, y + 23, f"{value:.4f}", 12, TEXT))

    if chance is not None:
        for value, label in chance:
            x = bar_x(value)
            parts.append(line(x, top, x, height - 34, MUTED, 1, "3 3"))
            parts.append(text(x, height - 20, label, 10, MUTED, "middle"))

    if note:
        parts.append(text(left, height - 6, note, 11, MUTED))
    parts.append("</svg>")
    out.write_text("".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=Path(__file__).parent / "logs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    points = []
    with open(args.logs / "val_log.csv") as fh:
        for row in csv.DictReader(fh):
            points.append((int(row["step"]), float(row["val_loss"])))
    val_loss_chart(points, (3478, 2.4366), args.out / "val-loss.svg")

    bar_chart(
        "Optimizer screen: 30M params, 50M tokens, step-700 loss",
        "one change per arm, identical seed and token budget",
        [
            ("AdamW (baseline)", 3.8041, MUTED),
            ("Muon", 3.5937, ACCENT),
            ("Muon + QK-Norm + soft-cap", 3.5103, ACCENT_DARK),
        ],
        args.out / "optimizer-ablation.svg",
        y_floor=3.4,
        y_step=3.9,
        note="y-axis starts at 3.4 to show differences; absolute values labeled",
    )
    bar_chart(
        "Architecture screen: 763 steps, step-700 loss",
        "same data and seed as the control arm",
        [
            ("Control (Muon + QK-Norm)", 3.4924, ACCENT_DARK),
            ("Looped depth (2 passes)", 3.5012, MUTED),
            ("Thin and deep (10L x 320)", 3.6969, MUTED),
            ("Grouped-query attention", 3.5031, MUTED),
        ],
        args.out / "architecture-ablation.svg",
        y_floor=3.4,
        y_step=3.9,
        note="no arm beat control by the pre-set 0.02 promotion margin",
    )
    bar_chart(
        "Zero-shot benchmarks: KALIA v0.1.2 (500 samples)",
        "lm-evaluation-harness, 0-shot; dashed lines mark chance level",
        [
            ("PIQA (acc)", 61.4, ACCENT),
            ("ARC-Easy (acc)", 45.8, ACCENT),
            ("HellaSwag (acc_norm)", 36.8, ACCENT),
            ("WinoGrande (acc)", 50.2, ACCENT),
        ],
        args.out / "benchmarks.svg",
        y_floor=0,
        y_step=80,
        chance=[(50, "chance 50"), (25, "chance 25")],
        note="standard error about 2.2 points at this sample size",
    )
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
