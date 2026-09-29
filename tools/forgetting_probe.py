"""Measure forgetting: score a checkpoint on the frozen probe and append to a ledger.

This is CL-0, the instrument every other continual-learning mechanism is tested
against. It answers one question -- *has this checkpoint lost ground on data it
was originally trained on?* -- which nothing in the project can currently answer.

The retention metric is deliberately not the loss itself. Absolute probe loss
depends on the probe's difficulty, which is fixed, so what matters is the drift
from the best score ever recorded on it:

    retention = probe_loss - best_probe_loss      (0 = best ever, >0 = regression)

Usage:
    python tools/forgetting_probe.py --ckpt out/ckpt.pt --step 1746
    python tools/forgetting_probe.py --ledger data/probe/forgetting.jsonl
"""

from __future__ import annotations

import argparse
import json
import math

import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import TokenDataset  # noqa: E402
from eval_reversibility import load_model  # noqa: E402


@torch.no_grad()
def probe_loss(model, dataset, batches: int, batch_size: int, seed: int, device) -> float:
    generator = torch.Generator().manual_seed(seed)
    total, tokens = 0.0, 0
    model.eval()
    for _ in range(batches):
        x, y = dataset.get_batch(batch_size, device, generator)
        _, loss = model(x, y)
        total += loss.item() * y.numel()
        tokens += int(y.numel())
    return total / tokens


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--step", type=int, default=None, help="override the step in the checkpoint")
    parser.add_argument("--probe", type=Path, default=Path("data/probe/val_forget.bin"))
    parser.add_argument("--ledger", type=Path, default=Path("data/probe/forgetting.jsonl"))
    parser.add_argument("--batches", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--commit", default=None)
    parser.add_argument("--no-bpb", action="store_true",
                        help="skip bytes-per-token measurement (avoids needing tiktoken)")
    args = parser.parse_args()

    rows = read_ledger(args.ledger)
    best = min((r["probe_loss"] for r in rows), default=None)

    if args.ckpt is None:
        # Report-only mode: print the curve and exit.
        print(f"{'step':>8} {'probe_loss':>11} {'drift':>9} {'note':>12}")
        for r in rows:
            drift = r["probe_loss"] - min(x["probe_loss"] for x in rows)
            print(f"{r['step']:>8} {r['probe_loss']:>11.4f} {drift:>+9.4f} {r.get('note',''):>12}")
        if rows:
            print(f"\nbest ever: {min(x['probe_loss'] for x in rows):.4f} at step "
                  f"{min(rows, key=lambda x: x['probe_loss'])['step']}")
        else:
            print("\nledger is empty -- run with --ckpt to record the first point")
        return

    device = torch.device(args.device)
    model = load_model(args.ckpt, device)
    dataset = TokenDataset(args.probe, model.cfg.context_len)
    loss = probe_loss(model, dataset, args.batches, args.batch_size, args.seed, device)
    step = args.step
    if step is None:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        step = int(ckpt["step"])

    # bits-per-byte, measured rather than assumed. This line used to divide by a
    # hardcoded 4.4086, which is not the bytes-per-token of our corpus: v0.2.0
    # measured 3.3775. Every bpB the CL-0 ledger had ever reported was therefore
    # ~30% too low, silently, and the ledger is the artifact Steps 2-4 are scored
    # against. eval_val.py already measures this correctly, so use that rather than
    # carrying a constant that goes stale the moment the mixture changes.
    bpt = None
    if not args.no_bpb:
        try:
            import tiktoken

            from eval_val import bytes_per_token

            bpt = bytes_per_token(dataset, tiktoken.get_encoding("gpt2"), 4, args.batch_size, args.seed)
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"note: could not measure bytes-per-token ({exc}); omitting bpB")

    drift = loss - best if best is not None else 0.0
    record = {
        "step": step,
        "probe_loss": round(loss, 6),
        "probe_tokens": args.batches * args.batch_size * model.cfg.context_len,
        "batches": args.batches,
        "seed": args.seed,
        "drift_from_best": round(drift, 6) if best is not None else None,
        "bpB": round(loss / math.log(2) / bpt, 6) if bpt else None,
        "bytes_per_token": round(bpt, 6) if bpt else None,
        "commit": args.commit,
        "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))
    if best is not None:
        verdict = "REGRESSION" if drift > 0.01 else "ok"
        print(f"\ndrift from best ever: {drift:+.4f} nats  [{verdict}]")


if __name__ == "__main__":
    main()
