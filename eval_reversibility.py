"""Entry-exit asymmetry ("Abhimanyu gap") for KALIA checkpoints.

A left-to-right trained model can "enter" fluent text but may be unable to
"exit" by processing it in reverse — the computational form of the Chakravyuha
episode. This module measures forward vs reversed-token NLL on held-out text;
closing the gap is the job of chunk-preserving reverse training.

Related literature: reversal curse (Berglund et al. 2023), reverse training
(Golovneva et al. 2024), reverse modeling as a data-quality signal
(arXiv 2410.09817).
"""

import argparse
import json
import math
from pathlib import Path

import torch

from model import GPT, GPTConfig


def load_model(ckpt_path: str | Path, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = GPTConfig(**ckpt["config"]["model"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def _nll_on_ids(model: GPT, ids: list[int], device: torch.device) -> tuple[float, int]:
    context_len = model.cfg.context_len
    ids = ids[: context_len + 1]
    if len(ids) < 2:
        return 0.0, 0
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    _, loss = model(x, y)
    return loss.item() * y.numel(), y.numel()


@torch.no_grad()
def forward_and_reverse_nll(
    model: GPT, encoder, sentences: list[str], device: torch.device | None = None
) -> dict:
    """Token-weighted forward NLL, reversed-token NLL, and their gap."""
    device = device or next(model.parameters()).device
    fwd_total, rev_total, tokens = 0.0, 0.0, 0
    per_sentence = []
    for sentence in sentences:
        ids = encoder.encode_ordinary(sentence)
        fwd_sum, n = _nll_on_ids(model, ids, device)
        rev_sum, _ = _nll_on_ids(model, list(reversed(ids)), device)
        if n == 0:
            continue
        fwd_total += fwd_sum
        rev_total += rev_sum
        tokens += n
        per_sentence.append(
            {
                "sentence": sentence,
                "forward_nll": round(fwd_sum / n, 4),
                "reverse_nll": round(rev_sum / n, 4),
                "quality_score": round((fwd_sum - rev_sum) / n, 4),
            }
        )
    forward = fwd_total / max(1, tokens)
    reverse = rev_total / max(1, tokens)
    return {
        "forward_nll": round(forward, 4),
        "reverse_nll": round(reverse, 4),
        "abhimanyu_gap": round(reverse - forward, 4),
        "per_sentence": per_sentence,
    }


def render_report(result: dict) -> str:
    lines = [
        "# Entry-Exit Asymmetry Report (Abhimanyu Gap)",
        "",
        f"- Forward NLL (can enter): {result['forward_nll']}",
        f"- Reverse-token NLL (can exit): {result['reverse_nll']}",
        f"- **Abhimanyu gap (reverse - forward): {result['abhimanyu_gap']}**",
        f"- Perplexity ratio: {math.exp(result['abhimanyu_gap']) if result['abhimanyu_gap'] < 50 else float('inf'):.3f}",
        "",
        "| Sentence | Forward | Reverse | Quality score (fwd-rev) |",
        "|---|---|---|---|",
    ]
    for row in result["per_sentence"]:
        lines.append(
            f"| {row['sentence'][:60]} | {row['forward_nll']} | "
            f"{row['reverse_nll']} | {row['quality_score']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--sentences", type=Path, default=Path("eval/probe_sentences.json"))
    parser.add_argument("--out", type=Path, default=Path("out/eval/reversibility.md"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    model = load_model(args.ckpt, torch.device(args.device))
    sentences = json.loads(args.sentences.read_text())["sentences"]
    result = forward_and_reverse_nll(model, encoder, sentences)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(result))
    print(f"report written to {args.out}")
    print(
        f"forward {result['forward_nll']} | reverse {result['reverse_nll']} | "
        f"gap {result['abhimanyu_gap']}"
    )


if __name__ == "__main__":
    main()
