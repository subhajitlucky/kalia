"""X17's 2x2: separate the mask's effect from the train/eval mismatch.

X17 scored the treatment **+0.0444** worse than its paired control, at both
seeds, 6x the paired spread. Both arms were evaluated with full causal
attention, so the control was trained and tested matched and the treatment was
tested in a regime it had never trained in. The registration warned about it --
"the eval must use the same masking as training to be meaningful" -- in its pass
branch, and the experiment never applied it.

`tools/kalia_x17_eval_kernel.py` scores all four archived checkpoints under both
evaluation settings and writes `mask2x2.json`. This reads it and computes the
decomposition, because the interesting quantity is not any single cell:

    interaction = (treatment - control | matched) - (treatment - control | mismatched)

A large negative interaction means the recorded penalty was the mismatch, not
the mechanism.

Usage:
    python tools/x17_mask2x2.py [--file docs/results/x17/mask2x2.json]

Exit codes describe computability, never outcome: `0` computed, `2` unreadable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "docs" / "results" / "x17" / "mask2x2.json"

SEEDS = ("1338", "1339")
CONTROL, TREATMENT = "control", "treatment"
DM1_BAR = -0.010  # the registered X17 bar, for reference only


def _same_sign(values: list[float]) -> bool:
    return all(v > 0 for v in values) or all(v < 0 for v in values)


def cell(cells: dict, arm: str, seed: str, masked: bool) -> float:
    key = f"{arm}-s{seed}|eval_{'masked' if masked else 'unmasked'}"
    if key not in cells:
        raise KeyError(f"{key} missing from the manifest (have {sorted(cells)})")
    return cells[key]


def compute(manifest: dict) -> dict:
    cells = manifest["cells"]
    deltas = {
        (seed, masked): cell(cells, TREATMENT, seed, masked)
        - cell(cells, CONTROL, seed, masked)
        for seed in SEEDS
        for masked in (False, True)
    }
    mean = {
        masked: sum(deltas[(s, masked)] for s in SEEDS) / len(SEEDS)
        for masked in (False, True)
    }
    # What applying the mask at eval does to a model that never trained with it.
    control_shift = sum(
        cell(cells, CONTROL, s, True) - cell(cells, CONTROL, s, False) for s in SEEDS
    ) / len(SEEDS)
    return {
        "deltas": {f"s{s}|{'masked' if m else 'unmasked'}": d for (s, m), d in deltas.items()},
        "mean_mismatched": mean[False],
        "mean_matched": mean[True],
        "interaction": mean[True] - mean[False],
        "control_eval_mask_shift": control_shift,
        # Concordance is agreement in sign, not a direction: both rows are
        # concordant here and they point opposite ways, which is the finding.
        "matched_concordant": _same_sign([deltas[(s, True)] for s in SEEDS]),
        "mismatched_concordant": _same_sign([deltas[(s, False)] for s in SEEDS]),
        "matched_clears_registered_bar": mean[True] <= DM1_BAR,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=DEFAULT)
    args = parser.parse_args()

    if not args.file.exists():
        print(f"cannot compute: {args.file} not found -- has the 2x2 kernel been archived?")
        return 2
    try:
        v = compute(json.loads(args.file.read_text()))
    except (KeyError, TypeError, ValueError) as exc:
        print(f"cannot compute: {exc}")
        return 2

    print(f"X17 2x2 — {args.file}")
    print("\n  paired deltas (treatment - control), per seed:")
    for key, d in v["deltas"].items():
        print(f"    {key:22s} {d:+.4f}")
    print()
    print(f"  mismatched (as X17 measured)   mean {v['mean_mismatched']:+.4f}"
          f"   concordant across seeds: {v['mismatched_concordant']}")
    print(f"  matched (train and eval agree) mean {v['mean_matched']:+.4f}"
          f"   concordant across seeds: {v['matched_concordant']}")
    print(f"  interaction (matched - mismatched) {v['interaction']:+.4f}")
    print(f"  control's own shift under eval-masking {v['control_eval_mask_shift']:+.4f}")
    print()
    print(f"  registered bar, for reference: DM-1 mean <= {DM1_BAR:+.3f}; "
          f"matched mean clears it: {v['matched_clears_registered_bar']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
