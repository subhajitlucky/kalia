"""Split token shards into a proportional validation set and an interleaved train set.

Usage (inside the Kaggle prep notebook):
    python mix_bins.py \
        --shards "data/tinystories.bin:20,data/smollm_fineweb_edu.bin:60,
                  data/cosmopedia.bin:15,data/stack_smol.bin:5" \
        --val-tokens 10000000 --train-tokens 2500000000 \
        --val-out data/val.bin --train-out data/train.bin
"""

import argparse
import json
from pathlib import Path

import numpy as np


def mix_shards(
    shards: list[tuple[Path, float]],
    val_out: Path,
    train_out: Path,
    val_tokens: int,
    train_tokens: int,
    block_tokens: int = 1_000_000,
) -> dict:
    """Write proportional val.bin, then interleaved train.bin from the remainder."""
    paths = [Path(p) for p, _ in shards]
    ratios = np.array([r for _, r in shards], dtype=np.float64)
    ratios = ratios / ratios.sum()

    remaining = np.array([p.stat().st_size // 2 for p in paths], dtype=np.int64)
    val_taken = np.zeros(len(paths), dtype=np.int64)

    # --- validation: proportional, taken from the start of each shard ---
    handles = [open(p, "rb") for p in paths]
    written_val = 0
    with open(val_out, "wb") as out:
        while written_val < val_tokens:
            progressed = False
            for i, handle in enumerate(handles):
                want = min(
                    int(block_tokens * ratios[i]),
                    val_tokens - written_val,
                    remaining[i] - val_taken[i],
                )
                if want <= 0:
                    continue
                data = handle.read(want * 2)
                out.write(data)
                written_val += len(data) // 2
                val_taken[i] += len(data) // 2
                progressed = True
            if not progressed:
                break
    for handle in handles:
        handle.close()

    # --- training: interleave the remainder proportionally ---
    handles = [open(p, "rb") for p in paths]
    for i, handle in enumerate(handles):
        handle.seek(int(val_taken[i]) * 2)
    written_train = 0
    with open(train_out, "wb") as out:
        while written_train < train_tokens:
            progressed = False
            for i, handle in enumerate(handles):
                want = min(
                    int(block_tokens * ratios[i]),
                    train_tokens - written_train,
                    remaining[i] - val_taken[i],
                )
                if want <= 0:
                    continue
                data = handle.read(want * 2)
                out.write(data)
                written_train += len(data) // 2
                val_taken[i] += len(data) // 2
                progressed = True
            if not progressed:
                break
    for handle in handles:
        handle.close()

    return {
        "val_tokens": written_val,
        "train_tokens": written_train,
        "shards": {
            str(p): {"ratio": float(r), "used_tokens": int(val_taken[i])}
            for i, (p, r) in enumerate(shards)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, help="comma list of path:ratio")
    parser.add_argument("--val-tokens", type=int, default=10_000_000)
    parser.add_argument("--train-tokens", type=int, default=2_500_000_000)
    parser.add_argument("--val-out", type=Path, default=Path("val.bin"))
    parser.add_argument("--train-out", type=Path, default=Path("train.bin"))
    parser.add_argument("--block-tokens", type=int, default=1_000_000)
    parser.add_argument("--meta", type=Path, default=None)
    args = parser.parse_args()

    shards = []
    for item in args.shards.split(","):
        path, ratio = item.strip().rsplit(":", 1)
        shards.append((Path(path.strip()), float(ratio)))

    stats = mix_shards(
        shards,
        args.val_out,
        args.train_out,
        args.val_tokens,
        args.train_tokens,
        args.block_tokens,
    )
    print(json.dumps(stats, indent=2))
    if args.meta is not None:
        args.meta.parent.mkdir(parents=True, exist_ok=True)
        args.meta.write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
