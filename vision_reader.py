"""X54: reader-guided diffusion consistency.

The generator is fine-tuned so that its own samples are readable by the frozen
from-scratch captioner. This is a self-consistency objective between two of our
own modules; no external teacher, no pretrained vision weights, no distillation
from another model.
"""

from __future__ import annotations

import torch

from vision_captioner import build_batch
from vision_diffusion import IMAGE_SIZE, IN_CHANNELS


def sample_with_grad(diff, model, text_emb: torch.Tensor, steps: int = 20,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Differentiable DDIM sampling.

    The update is exactly ``Diffusion.sample``; the only difference is that no
    ``torch.no_grad()`` is used, so a reader loss on the returned image flows
    through every denoising step into the U-Net.
    """
    b = text_emb.shape[0]
    x = torch.randn(b, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE,
                    device=text_emb.device, generator=generator)
    ts = torch.linspace(diff.steps - 1, 0, steps,
                        device=text_emb.device).long()
    for i, t in enumerate(ts):
        tt = t.expand(b)
        eps = model(x, tt, text_emb)
        a = diff.alpha_bar[t]
        x0 = (x - (1.0 - a).sqrt() * eps) / a.sqrt()
        x0 = x0.clamp(-1.0, 1.0)
        if i + 1 < len(ts):
            a_next = diff.alpha_bar[ts[i + 1]]
            x = a_next.sqrt() * x0 + (1.0 - a_next).sqrt() * eps
        else:
            x = x0
    return x.clamp(-1.0, 1.0)


def reader_token_loss(base, encoder, images: torch.Tensor,
                      captions: list[str], enc, device) -> torch.Tensor:
    """Cross-entropy of the frozen captioner on the given images.

    ``images`` are continuous 32x32 scenes in [-1, 1]. The captioner's image
    encoder turns them into four prefix tokens; the frozen decoder is teacher-
    forced with the correct caption and the loss is on the caption tokens.
    """
    imgs_t, x, y, mask, attn = build_batch(images, captions, enc, device)
    prefix = encoder(imgs_t)
    _, loss = base(x, targets=y, loss_mask=mask, attn_mask=attn,
                   prefix_embeds=prefix)
    return loss
