"""Compute X17's registered verdict from the archived logs.

The registration (`2026-09-28-X17-doc-mask.md`, amended
`2026-10-18-X17-amendment-1.md`) fixes five bars. X21's verdict was assembled by
hand from its results file, and a verdict assembled by hand is one transcription
away from being wrong — so this computes all five from the artefacts the run
leaves behind, and prints them with the numbers that decided each.

Every input is a CSV or a JSON the run already writes. Nothing is parsed out of
the kernel log: DM-3 turns on step time, and `train_log.csv` records `elapsed_s`
directly, which is exact where the log is a wall of buffered text. The first two
attempts at DM-3 parsed throughput out of the kernel log and both were wrong for
reasons that had nothing to do with the experiment — the log is JSON-framed, and
`ablate.py`'s own prints are block-buffered while its subprocess writes directly,
so the arm markers do not delimit the arms at all.

  val_log-<arm>-s<seed>.csv     the validation curve       (DM-1, DM-1b, DM-2, DM-5)
  train_log-<arm>-s<seed>.csv   step, loss, lr, tokens, elapsed_s   (DM-3)
  run_manifest.json             mask engagement measured before training  (DM-4)

Usage:
    python tools/x17_verdict.py                  # docs/results/x17
    python tools/x17_verdict.py --dir <path>

Exit codes describe whether the verdict could be computed, never whether the
experiment passed: `0` computed, `2` inputs missing or unparseable. A scientific
outcome must not masquerade as a build failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "docs" / "results" / "x17"

SEEDS = ("1338", "1339")
CONTROL, TREATMENT = "control", "treatment"
DM1_BAR = -0.010          # unchanged from the original registration
DM3_MAX_SLOWDOWN = 1.25   # treatment step time / control step time
DM2_MIN_POINTS = 3
DM2_MIN_TOTAL = 5


def _rows(path: Path) -> list[list[str]]:
    rows = path.read_text().strip().splitlines()
    if not rows:
        raise ValueError(f"{path.name}: empty")
    return [r.split(",") for r in rows]


def read_curve(path: Path) -> dict[int, float]:
    """`step,val_loss[,bpb]` -> {step: val_loss}."""
    rows = _rows(path)
    if not rows[0] or not rows[0][0].startswith("step"):
        raise ValueError(f"{path.name}: not a val log")
    return {int(r[0]): float(r[1]) for r in rows[1:] if len(r) >= 2}


def step_seconds(path: Path) -> float:
    """Steady-state seconds per step, excluding the first interval.

    `elapsed_s` is cumulative, so the difference between two rows over the steps
    between them is the real cost — warmup and the first fused kernels included
    in row one are excluded rather than averaged in, because they are not what
    DM-3 is bounding.
    """
    rows = _rows(path)
    if rows[0][:2] != ["step", "loss"] and "elapsed_s" not in rows[0]:
        raise ValueError(f"{path.name}: no elapsed_s column")
    body = rows[1:]
    if len(body) < 2:
        raise ValueError(f"{path.name}: too few rows to difference")
    first, last = body[0], body[-1]
    try:
        d_steps = int(last[0]) - int(first[0])
        d_time = float(last[4]) - float(first[4])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path.name}: unreadable step/elapsed_s ({exc})") from exc
    if d_steps <= 0:
        raise ValueError(f"{path.name}: no step span to measure")
    return d_time / d_steps


def compute(directory: Path = ARCHIVE) -> dict:
    directory = Path(directory)
    curves, costs = {}, {}
    for seed in SEEDS:
        for arm in (CONTROL, TREATMENT):
            val = directory / f"val_log-{arm}-s{seed}.csv"
            train = directory / f"train_log-{arm}-s{seed}.csv"
            for path in (val, train):
                if not path.exists():
                    raise FileNotFoundError(f"{path} not found -- has the run been archived?")
            curves[(arm, seed)] = read_curve(val)
            costs[(arm, seed)] = step_seconds(train)

    manifest_path = directory / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found -- DM-4 is measured there")
    engagement = float(json.loads(manifest_path.read_text())["dm4_mask_engagement"])

    per_seed = {}
    for seed in SEEDS:
        control, treatment = curves[(CONTROL, seed)], curves[(TREATMENT, seed)]
        shared = sorted(set(control) & set(treatment))
        per_seed[seed] = {
            "control_final": control[max(control)],
            "treatment_final": treatment[max(treatment)],
            "delta": treatment[max(treatment)] - control[max(control)],
            "points_ahead": sum(1 for s in shared if treatment[s] < control[s]),
            "points_total": len(shared),
            "control_s_per_step": costs[(CONTROL, seed)],
            "treatment_s_per_step": costs[(TREATMENT, seed)],
            "slowdown": costs[(TREATMENT, seed)] / costs[(CONTROL, seed)],
        }

    deltas = [v["delta"] for v in per_seed.values()]
    mean_delta = sum(deltas) / len(deltas)
    dm1 = mean_delta <= DM1_BAR
    dm1b = all(d < 0 for d in deltas)
    dm2 = all(v["points_ahead"] >= DM2_MIN_POINTS for v in per_seed.values())
    dm3 = all(v["slowdown"] <= DM3_MAX_SLOWDOWN for v in per_seed.values())
    return {
        "per_seed": per_seed,
        "dm1_mean_delta": mean_delta,
        "dm1": dm1,
        "dm1b": dm1b,
        "dm2": dm2,
        "dm3": dm3,
        "dm4_engagement": engagement,
        "dm4": engagement < 1.0,
        "dm5_spread": max(deltas) - min(deltas),
        # The registered decision rule: DM-1 and DM-2 pass -> promote.
        "promote": dm1 and dm1b and dm2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=ARCHIVE)
    args = parser.parse_args()

    try:
        v = compute(args.dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot compute a verdict: {exc}")
        return 2

    print(f"X17 doc-mask — archive {args.dir}")
    for seed, row in v["per_seed"].items():
        print(
            f"  seed {seed}: control {row['control_final']:.4f}  "
            f"treatment {row['treatment_final']:.4f}  delta {row['delta']:+.4f}  "
            f"ahead {row['points_ahead']}/{row['points_total']}  "
            f"step {row['control_s_per_step']:.3f}s -> {row['treatment_s_per_step']:.3f}s "
            f"(x{row['slowdown']:.3f})"
        )
    print()
    print(f"  DM-1   {'PASS' if v['dm1'] else 'FAIL'}  mean paired delta "
          f"{v['dm1_mean_delta']:+.4f} (bar {DM1_BAR:+.3f})")
    print(f"  DM-1b  {'PASS' if v['dm1b'] else 'FAIL'}  both deltas negative")
    print(f"  DM-2   {'PASS' if v['dm2'] else 'FAIL'}  treatment ahead at "
          f">={DM2_MIN_POINTS}/{DM2_MIN_TOTAL} points at both seeds")
    print(f"  DM-3   {'PASS' if v['dm3'] else 'FAIL'}  step-time slowdown "
          f"{max(r['slowdown'] for r in v['per_seed'].values()):.3f} (max {DM3_MAX_SLOWDOWN})")
    print(f"  DM-4   {'PASS' if v['dm4'] else 'FAIL'}  mask engagement "
          f"{v['dm4_engagement']:.4f} < 1.0")
    print(f"  DM-5   spread {v['dm5_spread']:.4f} (reported, no threshold)")
    print()
    print("  REGISTERED VERDICT:", "PROMOTE" if v["promote"] else "do not promote")
    return 0


if __name__ == "__main__":
    sys.exit(main())
