"""KALIA-Vision v0: a small conditional DDPM for 32x32 synthetic shapes.

The text condition is a frozen KALIA caption embedding (512-d). The diffusion
U-Net is trained from scratch; no pretrained vision weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 32
IN_CHANNELS = 3
COND_DIM = 128
TEXT_DIM = 512


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int = COND_DIM):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / max(1, half - 1)
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = COND_DIM):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond = nn.Linear(cond_dim, out_ch * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond(cond)[:, :, None, None].chunk(2, dim=1)
        h = h * (1.0 + scale) + shift
        return self.conv2(F.silu(self.norm2(h)))


class UNet(nn.Module):
    def __init__(self, cond_dim: int = COND_DIM):
        super().__init__()
        self.time = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.text = nn.Linear(TEXT_DIM, cond_dim)
        self.init = nn.Conv2d(IN_CHANNELS, 64, 3, padding=1)
        self.down1 = ConvBlock(64, 64, cond_dim)
        self.down2 = ConvBlock(64, 128, cond_dim)
        self.down3 = ConvBlock(128, 256, cond_dim)
        self.mid = ConvBlock(256, 256, cond_dim)
        self.up3 = ConvBlock(256 + 256, 128, cond_dim)
        self.up2 = ConvBlock(128 + 128, 64, cond_dim)
        self.up1 = ConvBlock(64 + 64, 64, cond_dim)
        self.out = nn.Conv2d(64, IN_CHANNELS, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                text_emb: torch.Tensor) -> torch.Tensor:
        cond = self.time(t) + self.text(text_emb)
        h1 = self.down1(self.init(x), cond)
        h2 = self.down2(F.max_pool2d(h1, 2), cond)
        h3 = self.down3(F.max_pool2d(h2, 2), cond)
        m = self.mid(F.max_pool2d(h3, 2), cond)
        u3 = self.up3(torch.cat([F.interpolate(m, scale_factor=2, mode="nearest"), h3], 1), cond)
        u2 = self.up2(torch.cat([F.interpolate(u3, scale_factor=2, mode="nearest"), h2], 1), cond)
        u1 = self.up1(torch.cat([F.interpolate(u2, scale_factor=2, mode="nearest"), h1], 1), cond)
        return self.out(u1)


class Diffusion:
    def __init__(self, steps: int = 1000, device: torch.device | str = "cpu",
                 image_size: int = IMAGE_SIZE):
        self.steps = steps
        self.device = torch.device(device)
        self.image_size = image_size
        betas = torch.linspace(1e-4, 0.02, steps, device=self.device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        a = self.alpha_bar[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1.0 - a).sqrt() * noise

    def loss(self, model: nn.Module, x0: torch.Tensor,
             text_emb: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(0, self.steps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps = model(x_t, t, text_emb)
        return F.mse_loss(eps, noise)

    @torch.no_grad()
    def sample(self, model: nn.Module, text_emb: torch.Tensor,
               steps: int = 100, generator: torch.Generator | None = None) -> torch.Tensor:
        model.eval()
        b = text_emb.shape[0]
        x = torch.randn(b, IN_CHANNELS, self.image_size, self.image_size,
                        device=text_emb.device, generator=generator)
        ts = torch.linspace(self.steps - 1, 0, steps, device=text_emb.device).long()
        for i, t in enumerate(ts):
            tt = t.expand(b)
            eps = model(x, tt, text_emb)
            a = self.alpha_bar[t]
            x0 = (x - (1.0 - a).sqrt() * eps) / a.sqrt()
            x0 = x0.clamp(-1.0, 1.0)
            if i + 1 < len(ts):
                a_next = self.alpha_bar[ts[i + 1]]
                x = a_next.sqrt() * x0 + (1.0 - a_next).sqrt() * eps
            else:
                x = x0
        return x.clamp(-1.0, 1.0)
