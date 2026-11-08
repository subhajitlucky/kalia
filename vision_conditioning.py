"""Text-conditioning embeddings for KALIA-Vision.

The v1 diffusion was trained with a **bidirectional masked mean** over the
caption tokens:

    h = base_text.encode(x, attn_mask=mask[:, None, None, :])
    pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)

The earlier see→draw→see cycle kernels instead used a **causal mean**
(``base_text.encode(ids).mean(1)``). X55 showed that the causal embedding is
far outside the diffusion's training distribution: with everything else fixed,
the held-out loop exact was 0.930 for the bidirectional embedding and 0.000 for
the causal one. The cosine similarity between the two is ~0.69.

Use ``mode="bidir"`` for the v1 diffusion. ``mode="causal"`` is kept only as
the control that documents the bug.
"""

from __future__ import annotations

import torch

MAX_TEXT_LEN = 32


@torch.no_grad()
def caption_embeddings(base_text, enc, captions: list[str],
                       mode: str = "bidir", device="cuda",
                       max_len: int = MAX_TEXT_LEN,
                       batch: int = 64) -> torch.Tensor:
    """Mean-pooled caption embeddings, shape ``(len(captions), 512)``.

    ``mode="bidir"`` reproduces the v1 diffusion's training condition.
    ``mode="causal"`` reproduces the earlier cycle-kernel bug.
    """
    assert mode in ("bidir", "causal"), mode
    out = []
    for start in range(0, len(captions), batch):
        chunk = captions[start:start + batch]
        ids = [enc.encode_ordinary(c)[:max_len] for c in chunk]
        width = max(len(x) for x in ids)
        x = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
        mask = torch.zeros((len(chunk), width), dtype=torch.bool, device=device)
        for i, seq in enumerate(ids):
            x[i, :len(seq)] = torch.tensor(seq, device=device)
            mask[i, :len(seq)] = True
        if mode == "bidir":
            attn = mask[:, None, None, :]
        else:
            causal = torch.ones(width, width, dtype=torch.bool,
                                device=device).tril()
            attn = causal.unsqueeze(0).unsqueeze(0) & mask[:, None, None, :]
        h = base_text.encode(x, attn_mask=attn)
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        out.append(pooled.float().cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def caption_embedding(base_text, enc, caption: str, mode: str = "bidir",
                      device="cuda", max_len: int = MAX_TEXT_LEN) -> torch.Tensor:
    """Single-caption convenience wrapper, shape ``(1, 512)``."""
    return caption_embeddings(base_text, enc, [caption], mode=mode,
                              device=device, max_len=max_len)
