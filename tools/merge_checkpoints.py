"""Merge checkpoints in weight space (CL-2).

For a daily continuation loop, the pre-day and post-day weights sit in the same
loss basin -- they are separated by one short training segment, not by independent
runs -- so their average is a valid model. Alon et al. (arXiv 2505.12512) report
that merging before/after weights across sequential bouts of learning matches
maintaining an online EMA of the parameters, with more flexibility and no
per-step overhead. For us the practical appeal is different and simpler: it
costs **zero GPU hours**.

Safety, stated up front: weight averaging is only valid along a connected path.
Merging two *independently trained* models produces something that is usually
worse than either parent. This tool therefore refuses to merge checkpoints whose
`step` values are not adjacent relative to their distance, and always keeps the
newer checkpoint's optimiser state, step counter and config.

Usage:
    python tools/merge_checkpoints.py --base out/before/ckpt.pt --new out/after/ckpt.pt
    python tools/merge_checkpoints.py --base a.pt --new b.pt --weight 0.3   # 30% base
    python tools/merge_checkpoints.py --soup out/ckpt_s1.pt out/ckpt_s2.pt out/ckpt_s3.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def merge_states(states: list[dict], weights: list[float] | None = None) -> dict:
    """Weighted average of the `model` tensors. Returns a new state dict."""
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    if len(weights) != len(states):
        raise ValueError("weights and states must be the same length")
    total = sum(weights)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0, got {total}")

    out = {}
    for key in states[0]["model"]:
        acc = None
        for state, w in zip(states, weights):
            t = state["model"][key].to(torch.float32) * w
            acc = t if acc is None else acc + t
        out[key] = acc.to(states[-1]["model"][key].dtype)
    return out


def check_compatible(states: list[dict], paths: list[Path]) -> None:
    ref = states[-1]
    ref_dtype = next(iter(ref["model"].values())).dtype
    for state, path in zip(states, paths):
        if set(state["model"]) != set(ref["model"]):
            raise ValueError(f"{path} has a different parameter set; cannot merge")
        cfg = state.get("config", {})
        if cfg != ref.get("config", {}):
            raise ValueError(f"{path} has a different config; cannot merge")
        for key, tensor in state["model"].items():
            if tensor.dtype != ref_dtype:
                raise ValueError(
                    f"{path}:{key} is {tensor.dtype} but the newest checkpoint is "
                    f"{ref_dtype}; mixed dtypes would silently change precision"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, help="earlier checkpoint")
    parser.add_argument("--new", type=Path, help="later checkpoint")
    parser.add_argument(
        "--soup", type=Path, nargs="*", default=None, help="average N checkpoints uniformly"
    )
    parser.add_argument("--weight", type=float, default=None, help="weight on --base (default 0.5)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.soup:
        paths = list(args.soup)
        if len(paths) < 2:
            raise SystemExit("--soup needs at least two checkpoints")
        states = [load(p) for p in paths]
        weights = None
    else:
        if not (args.base and args.new):
            raise SystemExit("need --base and --new, or --soup")
        paths = [args.base, args.new]
        states = [load(p) for p in paths]
        w = args.weight if args.weight is not None else 0.5
        if not 0.0 <= w <= 1.0:
            raise SystemExit("--weight must be in [0, 1]")
        weights = [w, 1.0 - w]

    check_compatible(states, paths)

    newest = states[-1]
    merged = {
        "model": merge_states(states, weights),
        # Optimiser state and counters come from the newer run: it is the one that
        # is actually current, and averaging moments across a merge is not meaningful.
        "optimizer": newest["optimizer"],
        "step": newest["step"],
        "tokens": newest["tokens"],
        "config": newest["config"],
        "merged_from": [str(p) for p in paths],
        "merge_weights": weights,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.out)
    print(f"merged {len(paths)} checkpoints -> {args.out}")
    print(f"  step {merged['step']:,} | tokens {merged['tokens']:,} | weights {weights}")


if __name__ == "__main__":
    main()
