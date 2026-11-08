"""KALIA-Vision v2: from-scratch image encoder + frozen KALIA caption decoder."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edit_compiler import apply_lora, load_lora, lora_state_dict

PREFIX_TOKENS = 4
PROMPT = "Describe: "
MAX_TEXT = 32
PAD_ID = 50256
NEWLINE_ID = 198
EOS_ID = 50256


class ImageEncoder(nn.Module):
    def __init__(self, n_embd: int = 512, tokens: int = PREFIX_TOKENS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GroupNorm(8, 256), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, tokens * n_embd)
        self.tokens = tokens
        self.n_embd = n_embd

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h = self.net(images).flatten(1)
        return self.fc(h).view(-1, self.tokens, self.n_embd)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_batch(images: torch.Tensor, captions: list[str], enc, device) -> tuple:
    """Build padded text tensors + causal mask with a 4-token image prefix."""
    prompts = [enc.encode_ordinary(PROMPT) for _ in captions]
    targets = [enc.encode_ordinary(c + "\n")[:MAX_TEXT] for c in captions]
    seqs = [p + t for p, t in zip(prompts, targets)]
    width = max(len(s) for s in seqs)
    b = len(seqs)
    x = torch.full((b, width - 1), PAD_ID, dtype=torch.long)
    y = torch.full((b, width - 1), -100, dtype=torch.long)
    loss_mask = torch.zeros((b, width - 1), dtype=torch.bool)
    token_mask = torch.zeros((b, PREFIX_TOKENS + width - 1), dtype=torch.bool)
    token_mask[:, :PREFIX_TOKENS] = True
    for i, (p, t, seq) in enumerate(zip(prompts, targets, seqs)):
        inp = seq[:-1]
        lab = seq[1:]
        n = len(inp)
        x[i, :n] = torch.tensor(inp)
        y[i, :n] = torch.tensor(lab)
        # prompt positions are not supervised
        loss_mask[i, :n] = False
        loss_mask[i, len(p) - 1:len(p) - 1 + len(t)] = True
        token_mask[i, PREFIX_TOKENS:PREFIX_TOKENS + n] = True
    # prepend prefix positions to labels/mask
    y_full = torch.full((b, PREFIX_TOKENS + width - 1), -100, dtype=torch.long)
    y_full[:, PREFIX_TOKENS:] = y
    mask_full = torch.zeros((b, PREFIX_TOKENS + width - 1), dtype=torch.bool)
    mask_full[:, PREFIX_TOKENS:] = loss_mask
    t = PREFIX_TOKENS + width - 1
    causal = torch.ones(t, t, dtype=torch.bool).tril()
    attn = causal.unsqueeze(0).unsqueeze(0) & token_mask.unsqueeze(1).unsqueeze(2)
    return (images.to(device), x.to(device), y_full.to(device),
            mask_full.to(device), attn.to(device))


@torch.no_grad()
def caption_image(base, encoder, image: torch.Tensor, enc, device,
                  max_new: int = MAX_TEXT) -> str:
    base.eval(); encoder.eval()
    prefix = encoder(image.to(device))
    idx = torch.tensor([enc.encode_ordinary(PROMPT)], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new):
        logits, _ = base(idx, prefix_embeds=prefix)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        tok = int(nxt.item())
        if tok in (NEWLINE_ID, EOS_ID):
            break
        out.append(tok)
        idx = torch.cat([idx, nxt], dim=1)
    return normalize(enc.decode(out))


def make_model(base_ckpt: Path, rank: int = 16, alpha: int = 16, device="cuda"):
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    from model import GPT, GPTConfig
    model = GPT(GPTConfig(**ckpt["config"]["model"]))
    load = model.load_state_dict(ckpt["model"], strict=False)
    assert not load.missing_keys and not load.unexpected_keys
    n = apply_lora(model, rank, alpha)
    model.to(device)
    encoder = ImageEncoder().to(device)
    return model, encoder, n
