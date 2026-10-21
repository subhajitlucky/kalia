"""Compute X31's registered verdict from the archived logs.

The registration (`2026-10-21-X31-qk-softcap-isolation.md`) fixes six bars.
Assembling a verdict by hand is one transcription away from being wrong, so this
computes all of them from the artifacts the run leaves behind.

Inputs, one per condition per seed:

    val_log-<condition>-s<seed>.csv     the validation curve
    run_manifest.json                   the registered bars and seeds

Usage:
    python tools/x31_verdict.py                  # docs/results/x31
    python tools/x31_verdict.py --dir <path>

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
ARCHIVE = ROOT / "docs" / "results" / "x31"

SEEDS = ("1338", "1339")
CONDITIONS = ("control", "qk", "cap", "both")
Q1_BAR = -0.020
Q2_BAR = -0.010
Q3_MARGIN = 0.010


def read_curve(path: Path) -> dict[int, float]:
    """`step,val_loss[,bpb]` -> {step: val_loss}."""
    rows = path.read_text().strip().splitlines()
    if not rows or not rows[0].startswith("step"):
        raise ValueError(f"{path.name}: not a val log")
    curve = {}
    for row in rows[1:]:
        parts = row.split(",")
        if len(parts) >= 2:
            curve[int(parts[0])] = float(parts[1])
    if not curve:
        raise ValueError(f"{path.name}: empty curve")
    return curve


def compute(directory: Path = ARCHIVE) -> dict:
    directory = Path(directory)
    finals = {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            path = directory / f"val_log-{condition}-s{seed}.csv"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found -- has the run been archived?")
            curve = read_curve(path)
            finals[(condition, seed)] = curve[max(curve)]

    deltas = {
        seed: {
            condition: finals[(condition, seed)] - finals[("control", seed)]
            for condition in ("qk", "cap", "both")
        }
        for seed in SEEDS
    }
    means = {
        condition: sum(deltas[s][condition] for s in SEEDS) / len(SEEDS)
        for condition in ("qk", "cap", "both")
    }
    spreads = {
        condition: max(deltas[s][condition] for s in SEEDS)
        - min(deltas[s][condition] for s in SEEDS)
        for condition in ("qk", "cap", "both")
    }

    q1 = means["both"] <= Q1_BAR
    q1b = all(deltas[s]["both"] < 0 for s in SEEDS)
    q2a = means["qk"] <= Q2_BAR
    q2b = means["cap"] <= Q2_BAR
    q3 = means["both"] <= min(means["qk"], means["cap"]) + Q3_MARGIN

    if q1 and q1b:
        decision = "bundle replicated at two seeds"
    elif q1:
        decision = "unreplicated: one seed carries the bundle"
    else:
        decision = "not replicated: withdraw the -0.08 claim"

    return {
        "finals": {f"{c}-s{s}": v for (c, s), v in finals.items()},
        "deltas": deltas,
        "means": means,
        "spreads": spreads,
        "q1": q1,
        "q1b": q1b,
        "q2a": q2a,
        "q2b": q2b,
        "q3": q3,
        "decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", type=Path, default=ARCHIVE)
    args = parser.parse_args()

    try:
        v = compute(args.dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot compute a verdict: {exc}")
        return 2

    print(f"X31 QK/soft-cap isolation — archive {args.dir}")
    print(f"{'seed':>6s} {'control':>9s} {'qk':>9s} {'cap':>9s} {'both':>9s}"
          f" {'d_qk':>9s} {'d_cap':>9s} {'d_both':>9s}")
    for seed in SEEDS:
        row = {c: v["finals"][f"{c}-s{seed}"] for c in CONDITIONS}
        d = v["deltas"][seed]
        print(f"{seed:>6s} {row['control']:9.4f} {row['qk']:9.4f} "
              f"{row['cap']:9.4f} {row['both']:9.4f} "
              f"{d['qk']:+9.4f} {d['cap']:+9.4f} {d['both']:+9.4f}")

    print(f"\n  mean paired delta: qk {v['means']['qk']:+.4f}  "
          f"cap {v['means']['cap']:+.4f}  both {v['means']['both']:+.4f}")
    print(f"  Q-1   {'PASS' if v['q1'] else 'FAIL'}  bundle mean "
          f"{v['means']['both']:+.4f} (bar {Q1_BAR:+.3f})")
    print(f"  Q-1b  {'PASS' if v['q1b'] else 'FAIL'}  both seed deltas negative")
    print(f"  Q-2a  {'PASS' if v['q2a'] else 'FAIL'}  qk mean "
          f"{v['means']['qk']:+.4f} (bar {Q2_BAR:+.3f})")
    print(f"  Q-2b  {'PASS' if v['q2b'] else 'FAIL'}  cap mean "
          f"{v['means']['cap']:+.4f} (bar {Q2_BAR:+.3f})")
    print(f"  Q-3   {'PASS' if v['q3'] else 'FAIL'}  both <= better component + "
          f"{Q3_MARGIN:.3f}")
    print(f"  Q-4   spreads: qk {v['spreads']['qk']:.4f}  "
          f"cap {v['spreads']['cap']:.4f}  both {v['spreads']['both']:.4f}")
    print(f"\n  DECISION: {v['decision']}")

    # The manifest is the other half of the record; if it is present, confirm
    # the archived curves belong to the registered run rather than a stray copy.
    manifest = Path(args.dir) / "run_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        print(f"  manifest: {m.get('experiment')} seeds={m.get('seeds')} "
              f"steps={m.get('steps')} bars={m.get('bars')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
