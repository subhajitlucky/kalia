"""High-precision held-out loss / bpB for a checkpoint on a token shard.

Fixed-seed batches make this deterministic: unlike the training loop's 20-batch
evals, this number does not move between runs and is suitable for publication.
"""

import argparse
import math
from pathlib import Path

import torch

from data import TokenDataset
from eval_reversibility import load_model


def bytes_per_token(dataset: TokenDataset, encoder, batches: int, batch_size: int, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    total_bytes = 0
    total_tokens = 0
    for _ in range(batches):
        x, _ = dataset.get_batch(batch_size, torch.device("cpu"), generator)
        for row in x:
            total_bytes += len(encoder.decode(row.tolist()).encode("utf-8"))
            total_tokens += int(row.numel())
    return total_bytes / total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--val-bin", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=Path("out/eval/val_loss.md"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    import tiktoken

    device = torch.device(args.device)
    model = load_model(args.ckpt, device)
    dataset = TokenDataset(args.val_bin, model.cfg.context_len)
    encoder = tiktoken.get_encoding("gpt2")

    generator = torch.Generator().manual_seed(args.seed)
    total, tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for _ in range(args.batches):
            x, y = dataset.get_batch(args.batch_size, device, generator)
            _, loss = model(x, y)
            total += loss.item() * y.numel()
            tokens += int(y.numel())
    mean_loss = total / tokens
    bpt = bytes_per_token(dataset, encoder, 10, args.batch_size, args.seed)
    bpb = mean_loss / math.log(2) / bpt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "# High-precision held-out loss\n\n"
        f"- Checkpoint: `{args.ckpt}`\n"
        f"- Batches: {args.batches} × {args.batch_size} × {model.cfg.context_len} "
        f"= {tokens:,} tokens (seed {args.seed}, deterministic)\n"
        f"- bytes/token: {bpt:.4f}\n"
        f"- **Val loss: {mean_loss:.4f}**\n"
        f"- **bpB: {bpb:.4f}**\n"
    )
    print(f"val loss {mean_loss:.4f} | bpB {bpb:.4f} | tokens {tokens:,}")


if __name__ == "__main__":
    main()
