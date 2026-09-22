"""Generate text from a trained KALIA checkpoint."""

import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig


def load_model(ckpt_path: str | Path, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"]["model"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def generate_text(
    ckpt_path: str | Path,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 200,
    encoder=None,
    device: torch.device | None = None,
) -> str:
    if encoder is None:
        import tiktoken

        encoder = tiktoken.get_encoding("gpt2")
    device = device or torch.device("cpu")
    model = load_model(ckpt_path, device)
    ids = encoder.encode_ordinary(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return encoder.decode(out[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    text = generate_text(
        args.ckpt,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=torch.device(args.device),
    )
    print(text)


if __name__ == "__main__":
    main()
