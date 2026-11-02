"""Score a checkpoint on the clean 2x2 composition benchmark.

The four cells are fixed by `composition_benchmark.py`:

- **S-S** seen sentence type, single change: the easy baseline.
- **S-P** seen sentence type, pair change: the real composition test.
- **U-S** unseen sentence type, single change: template/lexical generalization.
- **U-P** unseen sentence type, pair change: both at once.

The score is the mean suffix NLL over the candidate continuations; the lowest
NLL wins. The benchmark is a diagnostic, not a re-score of X40's registered
bars.

Usage:
    python eval_composition_benchmark.py --ckpt out/x40-pairs-s1338/ckpt.pt \
        --bench out/x40-bench/bench.jsonl --out out/x40-bench/pairs-s1338.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval_ip import accuracy_from_scores, score_candidates
from eval_reversibility import load_model
from intervention_pairs import InterventionDataset

CELL_ORDER = ("S-S", "S-P", "U-S", "U-P")
CELL_LABEL = {
    "S-S": "seen sentence / single change",
    "S-P": "seen sentence / PAIR change",
    "U-S": "unseen sentence / single change",
    "U-P": "unseen sentence / PAIR change",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rows", type=Path, default=None,
                        help="optional JSONL with per-record rows")
    args = parser.parse_args()

    import hashlib

    device = torch.device(args.device)
    model = load_model(args.ckpt, device)
    ds = InterventionDataset(args.bench, prefix_len=64, suffix_len=64)
    scores = score_candidates(model, ds, device, batch_size=args.batch_size)
    accuracy = accuracy_from_scores(ds, scores)

    by_cell = accuracy["by_cell"]
    print(f"checkpoint: {args.ckpt}")
    print(f"benchmark : {args.bench} ({len(ds.records)} records)")
    print(f"{'cell':4} {'n':>5} {'acc':>7} {'chance':>7} {'above':>8}  label")
    for cell in CELL_ORDER:
        bucket = by_cell.get(cell)
        if not bucket:
            print(f"{cell:4} missing")
            continue
        above = bucket["accuracy"] - bucket["chance"]
        print(
            f"{cell:4} {bucket['n']:5d} {bucket['accuracy']:7.4f} "
            f"{bucket['chance']:7.4f} {above:+8.4f}  {CELL_LABEL[cell]}"
        )
    print("\nby operation:")
    for key, bucket in sorted(accuracy["by_operation"].items()):
        print(f"  {key:20s} n={bucket['n']:4d} acc={bucket['accuracy']:.4f}")

    payload = {
        "ckpt": str(args.ckpt),
        "bench": str(args.bench),
        "bench_sha256": hashlib.sha256(args.bench.read_bytes()).hexdigest(),
        "n": len(ds.records),
        "by_cell": by_cell,
        "by_operation": accuracy["by_operation"],
        "singles": accuracy["seen"],
        "pairs": accuracy["unseen"],
        "overall": accuracy["overall"],
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}")
    if args.rows is not None:
        args.rows.parent.mkdir(parents=True, exist_ok=True)
        with open(args.rows, "w") as fh:
            for row in accuracy["rows"]:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"wrote {args.rows}")


if __name__ == "__main__":
    main()
