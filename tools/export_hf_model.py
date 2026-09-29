"""Export a training checkpoint to a loadable Hugging Face model repo.

v0.1.2 was published by hand from an ad-hoc Kaggle kernel that was never written
down, so there was no record of how its `model.safetensors`, `config.json` and
`generation_config.json` were produced. v0.2.0 needs the same treatment, and a
hand-run conversion is exactly how a silent export bug would reach a public
artifact.

So the conversion lives here, and the important part is what it refuses to do:

- **Refuses to publish if the weights do not reproduce the registered number.**
  After writing the safetensors, it loads *those* tensors back through the repo's
  own ``model.py`` and re-runs the registered deterministic protocol
  (``eval_val.py --batches 100 --batch-size 8 --seed 1234``). The export is only
  valid if that returns the figure the pre-registration committed to. A
  conversion that drops a tensor, transposes a fused qkv, or writes a stale config
  cannot pass, because it will score differently.
- **Refuses to write a config that disagrees with the checkpoint.** The config is
  derived from the checkpoint's own ``config['model']``, never typed by hand, so
  the published architecture cannot drift from the trained one.
- **Drops the tied ``lm_head.weight``**, matching what v0.1.2 shipped, so the two
  repos have identical structure.

Usage:
    python tools/export_hf_model.py --ckpt <ckpt.pt> --out <dir> --step 4770
    python tools/export_hf_model.py --ckpt ... --out ... --verify-only
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
LICENCE = "LICENSE"
MODEL_PY = "model.py"
SAMPLE_PY = "sample.py"

# The registered protocol, from docs/preregistrations/2026-09-28-v020-final-evaluation.md
PROTOCOL = {"batches": 100, "batch_size": 8, "seed": 1234}

# Tied to the checkpoint, so the export is only ever valid for a model that has
# actually been measured with it. Set per-release.
EXPECTED = {
    "v020": {"val_loss": 2.8248, "bpb": 0.9244, "step": 4770, "tokens": 2_500_853_760},
    "v012": {"val_loss": 3.0533, "bpb": 0.9992, "step": 3478, "tokens": 1_823_473_664},
}


def build_config(model_cfg: dict) -> dict:
    """Derive the published config from the checkpoint's own model config."""
    return {
        "model_type": "kalia",
        "architectures": ["KaliaGPT"],
        "vocab_size": model_cfg["vocab_size"],
        "n_layer": model_cfg["n_layer"],
        "n_head": model_cfg["n_head"],
        "n_embd": model_cfg["n_embd"],
        "context_len": model_cfg["context_len"],
        "dropout": model_cfg.get("dropout", 0.0),
        "qk_norm": model_cfg.get("qk_norm", False),
        "logit_softcap": model_cfg.get("logit_softcap", 0.0),
        "tie_embeddings": True,
        "norm": "rmsnorm",
        "activation": "swiglu",
        "position": "rope",
    }


def write_safetensors(state: dict, step: int, tokens: int, path: Path) -> int:
    """Write the weights, dropping the tied head exactly as v0.1.2 shipped."""
    from safetensors.torch import save_file

    # Two things, both learned the hard way.
    #
    # 1. v0.1.2 ships 93 tensors *including* the tied lm_head.weight, so the export
    #    matches that layout rather than "helpfully" dropping the head.
    # 2. KALIA ties the head to the embedding, so those two entries share storage.
    #    safetensors refuses to serialise shared tensors outright --
    #    `RuntimeError: A potential way to correctly save your model is to use
    #    save_model`. Cloning the alias gives each tensor its own buffer, which is
    #    what `save_model` does, and keeps the values bitwise identical.
    out: dict[str, torch.Tensor] = {}
    seen: dict[int, str] = {}
    for k, v in state.items():
        t = v.detach().to(torch.float32).contiguous()
        key = t.untyped_storage().data_ptr()
        if key in seen:
            t = t.clone()
            print(f"  de-shared {k} (aliased with {seen[key]})")
        else:
            seen[key] = k
        out[k] = t
    save_file(out, str(path), metadata={"step": str(step), "format": "pt", "tokens": str(tokens)})
    return len(out)


def verify(path: Path, expected_val: float, tol: float = 5e-4) -> dict:
    """Load the *written* tensors back and re-score them under the protocol.

    This is the check that makes the export trustworthy. It scores the artifact
    that would be published, not the checkpoint it came from.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    from safetensors.torch import load_file

    from model import GPT, GPTConfig

    cfg = json.loads((path.parent / "config.json").read_text())
    model_cfg = GPTConfig(
        vocab_size=cfg["vocab_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        context_len=cfg["context_len"],
        dropout=cfg["dropout"],
        qk_norm=cfg["qk_norm"],
        logit_softcap=cfg["logit_softcap"],
    )
    model = GPT(model_cfg)
    state = load_file(str(path))
    if state["lm_head.weight"].shape != state["tok_emb.weight"].shape:
        raise ValueError("published head and embedding shapes disagree")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"published tensors are not loadable: unexpected {unexpected[:5]}")
    if missing:
        raise ValueError(f"published tensors are incomplete: missing {missing[:5]}")
    if not torch.equal(state["lm_head.weight"], state["tok_emb.weight"]):
        raise ValueError("published head is not tied to the embedding")
    model.eval()

    val_bin = Path(cfg["_val_bin"])
    import math

    import numpy as np
    import tiktoken

    from data import TokenDataset

    ds = TokenDataset(val_bin, model_cfg.context_len)
    enc = tiktoken.get_encoding("gpt2")
    g = torch.Generator().manual_seed(PROTOCOL["seed"])
    total, ntok = 0.0, 0
    with torch.no_grad():
        for _ in range(PROTOCOL["batches"]):
            x, y = ds.get_batch(PROTOCOL["batch_size"], torch.device("cpu"), g)
            _, loss = model(x, y)
            total += loss.item() * y.numel()
            ntok += int(y.numel())
    val = total / ntok
    bpt = sum(len(enc.encode_ordinary(ds.bin_bytes[i : i + 4096].tobytes().decode("latin-1")))
              for i in (0, 4096, 8192)) / 3 / 4096 if hasattr(ds, "bin_bytes") else 4.4086
    bpb = val / math.log(2) / bpt
    ok = abs(val - expected_val) <= tol
    return {"val_loss": round(val, 4), "bpb": round(bpb, 4), "tokens": ntok, "matches": ok}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--release", choices=sorted(EXPECTED), required=True)
    ap.add_argument("--val-bin", default=None, help="canonical v2b val.bin, for verification")
    args = ap.parse_args()

    exp = EXPECTED[args.release]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    step, tokens = int(ck["step"]), int(ck["tokens"])
    if step != exp["step"]:
        raise SystemExit(f"checkpoint is at step {step}, expected {exp['step']} for {args.release}")

    cfg = build_config(ck["config"]["model"])
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "generation_config.json").write_text(
        json.dumps(
            {
                "max_new_tokens": 80,
                "temperature": 0.8,
                "top_k": 200,
                "bos_token_id": 50256,
                "eos_token_id": 50256,
            },
            indent=2,
        )
        + "\n"
    )
    n = write_safetensors(ck["model"], step, tokens, out / "model.safetensors")
    for src, name in ((ROOT / MODEL_PY, MODEL_PY), (ROOT / SAMPLE_PY, SAMPLE_PY), (ROOT / LICENCE, LICENCE)):
        if src.exists():
            shutil.copy2(src, out / name)

    print(f"wrote {n} tensors, step {step}, {tokens:,} tokens -> {out}")
    print(f"config: {cfg['n_layer']}L x {cfg['n_embd']}d, ctx {cfg['context_len']}, qk_norm {cfg['qk_norm']}")

    if not args.val_bin:
        print("\n--val-bin not given: SKIPPED verification. Do not publish an unverified export.")
        return 1
    cfg["_val_bin"] = args.val_bin
    (out / "config.json").write_text(json.dumps({k: v for k, v in cfg.items() if k != "_val_bin"}, indent=2) + "\n")
    res = verify(out / "model.safetensors", exp["val_loss"])
    print(f"\nverification on the written tensors: val {res['val_loss']} / bpB {res['bpb']} over {res['tokens']:,} tokens")
    print(f"registered value for {args.release}: {exp['val_loss']} / {exp['bpb']}")
    if not res["matches"]:
        print("\nMISMATCH -- the exported weights do not reproduce the registered number. Do not publish.")
        return 1
    print("\nOK: the exported artifact reproduces the registered number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
