"""Depth-expand a KALIA checkpoint by linear layer interpolation.

v0.3.1 has 10 layers. The 150M scale control uses 38 layers at the same width,
tokenizer and context. Each new layer is initialized by interpolating between
the two nearest source layers; the embedding, final norm and LM head are copied.
The expanded model is loaded strictly before it is saved, so a key/shape mistake
fails here rather than after a training run.

Usage:
    python tools/depth_expand.py --base-repo kalia-lm/kalia-v031 \
        --out out/kalia-v150-init.pt --layers 38
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import GPT, GPTConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-repo", default="kalia-lm/kalia-v031")
    parser.add_argument("--out", type=Path, default=Path("out/kalia-v150-init.pt"))
    parser.add_argument("--layers", type=int, default=38)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    hf_dir = out.parent / "hf"
    sf = hf_hub_download(args.base_repo, "model.safetensors",
                         local_dir=str(hf_dir))
    cf = hf_hub_download(args.base_repo, "config.json",
                         local_dir=str(hf_dir))
    raw = json.loads(Path(cf).read_text())
    cfg = {k: v for k, v in raw.items() if k in GPTConfig.__dataclass_fields__}
    old = GPT(GPTConfig(**cfg))
    load = old.load_state_dict(load_file(sf), strict=False)
    assert not load.missing_keys and not load.unexpected_keys, load
    old_sd = old.state_dict()
    old_n = int(cfg["n_layer"])
    new_n = int(args.layers)
    assert 0 < old_n < new_n, (old_n, new_n)

    new_sd = {
        "tok_emb.weight": old_sd["tok_emb.weight"].clone(),
        "norm_f.weight": old_sd["norm_f.weight"].clone(),
        "lm_head.weight": old_sd["lm_head.weight"].clone(),
    }
    block_keys = [k[len("blocks.0."):] for k in old_sd
                  if k.startswith("blocks.0.")]
    for i in range(new_n):
        pos = i * (old_n - 1) / (new_n - 1)
        lo = int(pos)
        hi = min(lo + 1, old_n - 1)
        alpha = pos - lo
        for suffix in block_keys:
            a = old_sd[f"blocks.{lo}.{suffix}"]
            b = old_sd[f"blocks.{hi}.{suffix}"]
            new_sd[f"blocks.{i}.{suffix}"] = (1.0 - alpha) * a + alpha * b

    new_cfg = dict(cfg)
    new_cfg["n_layer"] = new_n
    new = GPT(GPTConfig(**new_cfg))
    new.load_state_dict(new_sd, strict=True)
    params = sum(p.numel() for p in new.parameters())
    torch.save({
        "model": new_sd,
        "config": {"model": new_cfg},
        "step": 0,
        "tokens": 0,
        "depth_expand": {
            "old_layers": old_n,
            "new_layers": new_n,
            "method": "linear_interpolation",
            "base_repo": args.base_repo,
        },
    }, out)
    print(json.dumps({
        "out": str(out),
        "params": params,
        "old_layers": old_n,
        "new_layers": new_n,
        "bytes": out.stat().st_size,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
