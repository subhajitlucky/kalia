"""Deterministic evaluation probes for KALIA checkpoints.

Runs a fixed prompt set and a small held-out sentence set against any
checkpoint, producing a markdown report with samples and a loss / bits-per-byte
check. The prompt and sentence sets are frozen so every recipe change is
comparable across runs.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from model import GPT, GPTConfig


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[GPT, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = GPTConfig(**ckpt["config"]["model"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


def load_prompts(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text())
    prompts: list[str] = []
    for category in sorted(data["categories"]):
        prompts.extend(data["categories"][category])
    return prompts


def load_sentences(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text())["sentences"]


def generate(
    model: GPT,
    encoder,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int | None = 200,
    seed: int = 1337,
    device: torch.device | None = None,
) -> str:
    device = device or next(model.parameters()).device
    torch.manual_seed(seed)
    ids = encoder.encode_ordinary(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return encoder.decode(out[0].tolist())


@torch.no_grad()
def loss_and_bpb(
    model: GPT, encoder, sentences: list[str], device: torch.device | None = None
) -> tuple[float, float]:
    """Token-weighted mean loss and bits-per-byte over the sentence set."""
    device = device or next(model.parameters()).device
    context_len = model.cfg.context_len
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0
    for sentence in sentences:
        ids = encoder.encode_ordinary(sentence)[: context_len]
        if len(ids) < 2:
            continue
        x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
        _, loss = model(x, y)
        n = y.numel()
        total_loss += loss.item() * n
        total_tokens += n
        total_bytes += len(sentence.encode("utf-8"))
    mean_loss = total_loss / max(1, total_tokens)
    bpb = mean_loss / math.log(2) / (total_bytes / max(1, total_tokens))
    return mean_loss, bpb


def apply_loop_override(model: GPT, n_loops: int | None) -> GPT:
    """Override recurrent depth at evaluation time (Samyama depth scaling)."""
    if n_loops is not None:
        model.cfg.n_loops = int(n_loops)
    return model


def build_report(
    ckpt: dict,
    samples: dict[str, str],
    mean_loss: float,
    bpb: float,
    meta: dict,
) -> str:
    lines = [
        "# KALIA Evaluation Report",
        "",
        f"- Checkpoint step: {ckpt.get('step')}",
        f"- Tokens seen: {int(ckpt.get('tokens', 0)):,}",
        f"- Seed: {meta['seed']} | temperature: {meta['temperature']} | top-k: {meta['top_k']}",
        f"- Held-out sentence loss: **{mean_loss:.4f}** | bits-per-byte: **{bpb:.4f}**",
        "",
        "## Samples",
        "",
    ]
    for prompt, text in samples.items():
        lines += [f"### `{prompt}`", "", text, ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("eval/probe_prompts.json"))
    parser.add_argument("--sentences", type=Path, default=Path("eval/probe_sentences.json"))
    parser.add_argument("--out", type=Path, default=Path("out/eval/report.md"))
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-loops", type=int, default=None, help="override recurrent depth")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    import tiktoken

    device = torch.device(args.device)
    encoder = tiktoken.get_encoding("gpt2")
    model, ckpt = load_model(args.ckpt, device)
    apply_loop_override(model, args.n_loops)
    prompts = load_prompts(args.prompts)
    sentences = load_sentences(args.sentences)

    samples = {
        prompt: generate(
            model,
            encoder,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            device=device,
        )
        for prompt in prompts
    }
    mean_loss, bpb = loss_and_bpb(model, encoder, sentences, device)
    meta = {
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    report = build_report(ckpt, samples, mean_loss, bpb, meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"report written to {args.out}")
    print(f"held-out loss {mean_loss:.4f} | bpB {bpb:.4f}")


if __name__ == "__main__":
    main()
