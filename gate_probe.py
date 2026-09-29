"""Measure the Gated Residual gate opening on real text (X18, F-4).

F-4 is pre-registered as "mean gate value at the end > 0.05". The gate starts
near-closed by construction: ``w2.weight`` is zero-initialised and
``w2.bias = -4``, so ``sigmoid(-4) = 0.018`` at step 0. The question is whether
training opened it.

The measurement replays the real forward pass (no external positional
embedding -- RoPE is applied inside attention) and reads the gate exactly where
``Block.forward`` computes it, so the reported mean is the gate the model
actually trained with rather than a re-derivation of it.

This is a diagnostic. F-4 failing does not promote or reject anything on its
own; see the pre-registration's decision rule.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

INIT_GATE_MEAN = 1.0 / (1.0 + math.exp(4.0))  # sigmoid(-4): the designed near-closed start
F4_THRESHOLD = 0.05


@torch.no_grad()
def gate_stats(model, token_batches) -> dict:
    """Gate mean and dispersion per block, over the given batches.

    X18's F-4 was a *mean* threshold, and F-4 failed at 0.0192. But a mean is a
    first moment: it cannot distinguish a constant gate from one that is low on
    average and strongly variable. A gate sitting at 0.019 that ranges over
    0.001-0.5 by channel is doing substantial work while looking closed.

    So X19's G-0 measures the second moments as well, to test whether our own
    F-4 interpretation was wrong:

    - ``std_channel``: spread across the embedding dimension at a fixed input.
      Near zero means every channel is gated identically -- a constant
      reweighting, i.e. no data dependence at all.
    - ``std_input``: spread of the per-position channel-mean across positions and
      documents. Near zero means the gate ignores its input entirely.

    Both near zero => the gate is genuinely inert and F-4's reading stands.
    Either one large => the gate is data-dependent and F-4's threshold was the
    wrong statistic.
    """
    blocks = model.blocks
    n_gated = sum(1 for b in blocks if b.gated is not None)
    if n_gated == 0:
        raise RuntimeError("model has no GatedResidual blocks -- wrong checkpoint")

    totals = torch.zeros(len(blocks), dtype=torch.float64)
    sq_channel = torch.zeros(len(blocks), dtype=torch.float64)
    pos_means: list[list[torch.Tensor]] = [[] for _ in blocks]
    counts = 0
    n_pos = 0

    for x in token_batches:
        h = model.tok_emb(x)
        for _ in range(model.cfg.n_loops):
            for i, block in enumerate(blocks):
                if block.gated is not None:
                    g = block.gated
                    gate = torch.sigmoid(g.w2(F.silu(g.w1(h))))
                    # Spread across the embedding dim, per position.
                    totals[i] += gate.mean().item() * x.shape[0]
                    sq_channel[i] += gate.var(dim=-1, unbiased=False).mean().item() * x.shape[0]
                    # Per-position mean, so we can ask whether the gate responds
                    # to its input at all.
                    pos_means[i].append(gate.mean(dim=-1).reshape(-1).clone())
                pre = block.gated(h) if block.gated is not None else h
                h = h + block.attn(block.norm1(pre), None)
                h = h + block.mlp(block.norm2(h))
        counts += x.shape[0]
        n_pos += x.shape[1]

    per_block = (totals / counts).tolist()
    mean_per_block = torch.tensor(per_block, dtype=torch.float64)
    std_channel = torch.sqrt((sq_channel / counts).clamp_min(0)).tolist()
    std_input = []
    for i in range(len(blocks)):
        if not pos_means[i]:
            std_input.append(0.0)
            continue
        stacked = torch.cat(pos_means[i]).double()
        std_input.append(stacked.std(unbiased=False).item())

    return {
        "per_block": per_block,
        "gated_blocks": n_gated,
        "overall": sum(per_block) / len(per_block),
        "std_channel": std_channel,
        "std_input": std_input,
        "max_std_channel": max(std_channel) if std_channel else 0.0,
        "max_std_input": max(std_input) if std_input else 0.0,
    }


@torch.no_grad()
def gate_means(model, token_batches) -> dict:
    """Mean gate value per gated block and overall (the original F-4 measure)."""
    return gate_stats(model, token_batches)


def main() -> None:
    import argparse

    import tiktoken

    from eval_reversibility import load_model

    parser = argparse.ArgumentParser(description="X18 F-4 gate opening measurement")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--sentences", default="eval/probe_sentences.json")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    encoder = tiktoken.get_encoding("gpt2")
    ctx = model.cfg.context_len
    sentences = json.loads(Path(args.sentences).read_text())["sentences"]

    def encode(text: str) -> list[int]:
        return encoder.encode_ordinary(text)[:ctx]

    batches = []
    for sent in sentences:
        ids = encode(sent)
        if len(ids) >= 2:
            batches.append(torch.tensor(ids, dtype=torch.long, device=device)[None])

    # One full-length real context, to check the mean is not an artefact of the
    # short probe sentences.
    joined = encode(" ".join(sentences))
    if len(joined) > 2:
        batches.append(torch.tensor(joined, dtype=torch.long, device=device)[None])

    result = gate_stats(model, batches)
    result["batches"] = len(batches)
    result["init_mean"] = INIT_GATE_MEAN
    result["delta"] = result["overall"] - INIT_GATE_MEAN
    result["f4_threshold"] = F4_THRESHOLD
    result["f4_passes"] = bool(result["overall"] > F4_THRESHOLD)

    print("X18 F-4: gate opening")
    for i, v in enumerate(result["per_block"]):
        print(f"  block {i}: mean gate {v:.4f}")
    print(
        f"  overall {result['overall']:.4f} | init {result['init_mean']:.4f} "
        f"| delta {result['delta']:+.4f} | threshold {F4_THRESHOLD} "
        f"| F-4 {'PASS' if result['f4_passes'] else 'FAIL'}"
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
