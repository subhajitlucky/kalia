"""Carve a frozen replay shard out of an existing train.bin (CL-1).

Run as a Kaggle **CPU** kernel, so it costs no GPU quota.

Why a shard at all
------------------
Continual-learning replay is only meaningful if the replayed tokens are ones the
daily run would otherwise *not* see. If the shard is carved out of `train.bin`
and the daily loop keeps training on all of `train.bin`, the replay is
redundant: the same tokens arrive anyway, and nothing is protected.

So this script does two things, and the second matters as much as the first:

1. Writes `replay.bin` -- a set of contiguous blocks sampled across the file, so
   the shard is spread over the whole corpus and keeps the mixer's 4-corpus
   interleaving rather than being one contiguous region.
2. Writes `replay_manifest.json` recording the exact token ranges used.

The daily data pipeline must then **exclude those ranges** from its "new" feed.
That is the step that turns replay from a no-op into a safeguard, and the
manifest exists so the exclusion is verifiable rather than a claim in a doc.

Contiguous blocks (not uniformly random windows) because a block boundary may cut
a document, but with a handful of boundaries per hundred million tokens the cost
is negligible -- whereas random windows would destroy document structure
throughout, which is the defect we just fixed (I15).

Usage (on Kaggle):
    python make_replay_shard.py --train-bin /kaggle/input/.../train.bin --out-dir /kaggle/working/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DOC_SEPARATOR = 50256


def choose_blocks(total_tokens: int, target_tokens: int, n_blocks: int, seed: int) -> list[tuple[int, int]]:
    """Pick `n_blocks` contiguous, non-overlapping, evenly spread token ranges."""
    block = target_tokens // n_blocks
    if block <= 0:
        raise SystemExit("target too small for the requested number of blocks")
    if block * n_blocks > total_tokens:
        raise SystemExit("target exceeds the source file")
    stride = total_tokens // n_blocks
    rng = np.random.default_rng(seed)
    ranges = []
    for i in range(n_blocks):
        lo = i * stride
        # Jitter within the stride so blocks are spread, not on a rigid grid.
        offset = int(rng.integers(0, max(1, stride - block)))
        start = lo + offset
        ranges.append((start, block))
    return ranges


def align_to_documents(tokens: np.ndarray) -> np.ndarray:
    """Trim leading and trailing partial documents so every block is whole."""
    if tokens.size == 0 or tokens[0] != DOC_SEPARATOR:
        first = np.flatnonzero(tokens == DOC_SEPARATOR)
        tokens = tokens[first[0] :] if first.size else tokens
    if tokens.size and tokens[-1] != DOC_SEPARATOR:
        last = np.flatnonzero(tokens == DOC_SEPARATOR)
        if last.size:
            tokens = tokens[: last[-1] + 1]
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bin", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.10,
        help="fraction of train.bin to carve (10%% -> ~240M tokens of 2.4B)",
    )
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    src = np.memmap(args.train_bin, dtype=np.uint16, mode="r")
    total = len(src)
    target = int(total * args.fraction)
    print(f"source: {args.train_bin} | {total:,} tokens | target {target:,} ({args.fraction:.0%})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "replay.bin"
    ranges = choose_blocks(total, target, args.blocks, args.seed)

    kept = 0
    with open(out_path, "wb") as fh:
        for start, length in ranges:
            chunk = np.asarray(src[start : start + length], dtype=np.uint16)
            chunk = align_to_documents(chunk)
            if chunk.size:
                fh.write(chunk.tobytes())
                kept += chunk.size

    digest = hashlib.sha256()
    with open(out_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)

    manifest = {
        "purpose": "CL-1 replay shard; EXCLUDE these ranges from any future 'new data' feed",
        "source": str(args.train_bin),
        "source_tokens": total,
        "fraction": args.fraction,
        "seed": args.seed,
        "replay_tokens": kept,
        "blocks": [[int(s), int(l)] for s, l in ranges],
        "sha256": digest.hexdigest(),
        "bytes": out_path.stat().st_size,
    }
    (args.out_dir / "replay_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "blocks"}, indent=2))
    print(f"\ncarved {kept:,} tokens into {out_path}")
    print("The daily pipeline must skip these ranges, or replay protects nothing.")


if __name__ == "__main__":
    main()
