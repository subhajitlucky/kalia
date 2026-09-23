"""Generate identical prompts from two KALIA checkpoints, side by side.

Used to compare recipes at equal steps (e.g., AdamW baseline vs the
Muon+ recipe) with deterministic sampling.
"""

import argparse
from pathlib import Path

import torch

from eval_probes import generate, load_model, load_prompts


def compare(
    ckpt_a: Path,
    ckpt_b: Path,
    prompts: list[str],
    encoder,
    out_path: Path,
    label_a: str = "A",
    label_b: str = "B",
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int | None = 200,
    seed: int = 1337,
    device: torch.device | None = None,
) -> Path:
    device = device or torch.device("cpu")
    model_a, meta_a = load_model(ckpt_a, device)
    model_b, meta_b = load_model(ckpt_b, device)

    lines = [
        "# KALIA Model Comparison",
        "",
        f"- {label_a}: step {meta_a.get('step')} ({int(meta_a.get('tokens', 0)):,} tokens)",
        f"- {label_b}: step {meta_b.get('step')} ({int(meta_b.get('tokens', 0)):,} tokens)",
        f"- seed {seed} | temperature {temperature} | top-k {top_k}",
        "",
    ]
    for prompt in prompts:
        text_a = generate(
            model_a, encoder, prompt,
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            seed=seed, device=device,
        )
        text_b = generate(
            model_b, encoder, prompt,
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            seed=seed, device=device,
        )
        lines += [f"### `{prompt}`", "", f"**{label_a}:** {text_a}", "", f"**{label_b}:** {text_b}", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-a", type=Path, required=True)
    parser.add_argument("--ckpt-b", type=Path, required=True)
    parser.add_argument("--label-a", type=str, default="baseline")
    parser.add_argument("--label-b", type=str, default="upgraded")
    parser.add_argument("--prompts", type=Path, default=Path("eval/probe_prompts.json"))
    parser.add_argument("--out", type=Path, default=Path("out/eval/comparison.md"))
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    out = compare(
        args.ckpt_a,
        args.ckpt_b,
        load_prompts(args.prompts),
        encoder,
        args.out,
        label_a=args.label_a,
        label_b=args.label_b,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        device=torch.device(args.device),
    )
    print(f"comparison written to {out}")


if __name__ == "__main__":
    main()
