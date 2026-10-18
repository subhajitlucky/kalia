"""Compute X17's registered verdict from the archived logs.

The registration (`2026-09-28-X17-doc-mask.md`, amended
`2026-10-18-X17-amendment-1.md`) fixes five bars. X21's verdict was assembled by
hand from its results file, and a verdict assembled by hand is one transcription
away from being wrong — so this computes all five from the artefacts the run
leaves behind, and prints them with the numbers that decided each.

Both inputs are things the run already produces; nothing here is inferred:

  val_log-<arm>-s<seed>.csv   the per-arm validation curve (train.py writes it)
  <run>.log                   the kernel log, for step throughput (DM-3)
  run_manifest.json           for DM-4, the mask engagement measured before
                              training

Usage:
    python tools/x17_verdict.py                  # docs/results/x17
    python tools/x17_verdict.py --dir <path>

Exit codes describe whether the verdict could be computed, never whether the
experiment passed: a scientific outcome must not masquerade as a build failure.
`0` computed, `2` inputs missing or unparseable.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
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

TOKS = re.compile(r"step \d+/\d+ \| loss [\d.]+ \| lr [\deE.+-]+ \| ([\d,]+) tok/s")
ARM_MARK = re.compile(r"^ARM: (\S+)", re.M)


def read_curve(path: Path) -> dict[int, float]:
    """`step,val_loss[,bpb]` -> {step: val_loss}."""
    rows = path.read_text().strip().splitlines()
    assert rows and rows[0].startswith("step"), f"{path.name}: not a val log"
    out = {}
    for row in rows[1:]:
        cells = row.split(",")
        if len(cells) >= 2:
            out[int(cells[0])] = float(cells[1])
    return out


def throughput_by_arm(log_text: str) -> dict[str, float]:
    """Median tok/s per arm, segmented by ablate.py's `ARM: <name>` markers."""
    markers = list(ARM_MARK.finditer(log_text))
    rates: dict[str, list[int]] = {}
    for i, m in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(log_text)
        chunk = log_text[m.end():end]
        hits = [int(v.replace(",", "")) for v in TOKS.findall(chunk)]
        if hits:
            rates[m.group(1)] = hits
    return {arm: statistics.median(v) for arm, v in rates.items()}


def compute(directory: Path = ARCHIVE) -> dict:
    directory = Path(directory)
    curves = {}
    for seed in SEEDS:
        for arm in (CONTROL, TREATMENT):
            path = directory / f"val_log-{arm}-s{seed}.csv"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found -- has the run been archived?")
            curves[(arm, seed)] = read_curve(path)

    logs = sorted(directory.glob("*.log"))
    if not logs:
        raise FileNotFoundError(f"no kernel log in {directory} -- DM-3 needs its throughput")
    rates = throughput_by_arm(logs[0].read_text())
    if not rates:
        raise ValueError(f"{logs[0].name}: no `ARM:` markers with tok/s lines")

    manifest_path = directory / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found -- DM-4 is measured there")
    engagement = float(json.loads(manifest_path.read_text())["dm4_mask_engagement"])

    per_seed = {}
    for seed in SEEDS:
        control = curves[(CONTROL, seed)]
        treatment = curves[(TREATMENT, seed)]
        shared = sorted(set(control) & set(treatment))
        ahead = sum(1 for s in shared if treatment[s] < control[s])
        c_tps = rates.get(f"{CONTROL}-s{seed}")
        t_tps = rates.get(f"{TREATMENT}-s{seed}")
        per_seed[seed] = {
            "control_final": control[max(control)],
            "treatment_final": treatment[max(treatment)],
            "delta": treatment[max(treatment)] - control[max(control)],
            "points_ahead": ahead,
            "points_total": len(shared),
            # step time is inverse throughput, so this is the slowdown factor
            "slowdown": (c_tps / t_tps) if (c_tps and t_tps) else None,
        }

    deltas = [v["delta"] for v in per_seed.values()]
    mean_delta = sum(deltas) / len(deltas)
    return {
        "per_seed": per_seed,
        "dm1_mean_delta": mean_delta,
        "dm1": mean_delta <= DM1_BAR,
        "dm1b": all(d < 0 for d in deltas),
        "dm2": all(v["points_ahead"] >= DM2_MIN_POINTS for v in per_seed.values()),
        "dm3_ratios": {s: v["slowdown"] for s, v in per_seed.items()},
        "dm3": all(r is not None and r <= DM3_MAX_SLOWDOWN for r in
                   (v["slowdown"] for v in per_seed.values())),
        "dm4_engagement": engagement,
        "dm4": engagement < 1.0,
        "dm5_spread": max(deltas) - min(deltas),
        # The registered decision rule: DM-1 and DM-2 pass -> promote.
        "promote": mean_delta <= DM1_BAR and all(d < 0 for d in deltas)
        and all(v["points_ahead"] >= DM2_MIN_POINTS for v in per_seed.values()),
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
        slow = f"{row['slowdown']:.3f}" if row["slowdown"] else "n/a"
        print(
            f"  seed {seed}: control {row['control_final']:.4f}  "
            f"treatment {row['treatment_final']:.4f}  delta {row['delta']:+.4f}  "
            f"ahead {row['points_ahead']}/{row['points_total']}  slowdown {slow}"
        )
    print()
    print(f"  DM-1   {'PASS' if v['dm1'] else 'FAIL'}  mean paired delta "
          f"{v['dm1_mean_delta']:+.4f} (bar {DM1_BAR:+.3f})")
    print(f"  DM-1b  {'PASS' if v['dm1b'] else 'FAIL'}  both deltas negative")
    print(f"  DM-2   {'PASS' if v['dm2'] else 'FAIL'}  treatment ahead at "
          f">={DM2_MIN_POINTS}/{DM2_MIN_TOTAL} points at both seeds")
    print(f"  DM-3   {'PASS' if v['dm3'] else 'FAIL'}  step-time slowdown "
          f"{', '.join(f'{s}:{r:.3f}' for s, r in v['dm3_ratios'].items() if r)} "
          f"(max {DM3_MAX_SLOWDOWN})")
    print(f"  DM-4   {'PASS' if v['dm4'] else 'FAIL'}  mask engagement "
          f"{v['dm4_engagement']:.4f} < 1.0")
    print(f"  DM-5   spread {v['dm5_spread']:.4f} (reported, no threshold)")
    print()
    print("  REGISTERED VERDICT:", "PROMOTE" if v["promote"] else "do not promote")
    return 0


if __name__ == "__main__":
    sys.exit(main())
