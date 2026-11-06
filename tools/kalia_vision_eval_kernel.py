"""Kaggle GPU kernel: evaluate KALIA-Vision v0.

Loads the trained diffusion checkpoint, samples the held-out test captions,
and trains a small from-scratch attribute classifier (shape/color/position)
to measure W-1..W-3. Saves a sample grid.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

hits = sorted(Path("/kaggle/input").rglob("vision_diffusion.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))

from vision_diffusion import Diffusion, UNet  # noqa: E402
from synthetic_vision import COLORS, POSITIONS, SHAPES  # noqa: E402

data_hits = sorted(Path("/kaggle/input").rglob("test_images.npy"))
assert data_hits, "kalia-vision-data not attached"
data_dir = data_hits[0].parent
ckpt_hits = sorted(Path("/kaggle/input").rglob("ckpt.pt"))
ckpt_hits = [p for p in ckpt_hits if "vision" in str(p)]
assert ckpt_hits, "kalia-vision-train output not attached"
ckpt_path = ckpt_hits[0]
emb_hits = sorted(Path("/kaggle/input").rglob("text_emb.npz"))
assert emb_hits, "text_emb.npz missing"
emb_path = emb_hits[0]
print("ckpt:", ckpt_path, "emb:", emb_path, flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"

model = UNet().to(device)
payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(payload["model"])
model.eval()
print("checkpoint step:", payload.get("step"), flush=True)

z = np.load(emb_path)
test_emb = torch.from_numpy(z["test"]).float()
test_images = np.load(data_dir / "test_images.npy")
test_caps = [json.loads(line)["caption"] for line in
             (data_dir / "test_captions.jsonl").read_text().splitlines() if line.strip()]


def parse_caption(caption: str) -> tuple[int, int, int]:
    shape = next(i for i, s in enumerate(SHAPES) if s in caption)
    color = next(i for i, c in enumerate(COLORS) if c in caption)
    pos = next(i for i, p in enumerate(POSITIONS) if p in caption)
    return shape, color, pos


test_labels = np.array([parse_caption(c) for c in test_caps], dtype=np.int64)
train_images = np.load(data_dir / "train_images.npy")
train_caps = [json.loads(line)["caption"] for line in
              (data_dir / "train_captions.jsonl").read_text().splitlines() if line.strip()]
train_labels = np.array([parse_caption(c) for c in train_caps], dtype=np.int64)

# --- sample ---------------------------------------------------------------
out_dir = Path("/kaggle/working/vision")
out_dir.mkdir(parents=True, exist_ok=True)
diff = Diffusion(1000, device)
gen = []
with torch.no_grad():
    for start in range(0, len(test_emb), 128):
        emb = test_emb[start:start + 128].to(device)
        gen.append(diff.sample(model, emb, steps=100).cpu())
generated = torch.cat(gen, dim=0)
gen_np = ((generated.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
np.save(out_dir / "generated_test.npy", gen_np)
grid = gen_np[:64]
canvas = np.zeros((8 * 32, 8 * 32, 3), dtype=np.uint8)
for i in range(64):
    r, c = divmod(i, 8)
    canvas[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = grid[i]
Image.fromarray(canvas).resize((512, 512), Image.NEAREST).save(
    str(out_dir / "grid.png")
)
print("sampled", gen_np.shape, flush=True)


# --- attribute classifier -------------------------------------------------
class Cls(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.shape = nn.Linear(128, len(SHAPES))
        self.color = nn.Linear(128, len(COLORS))
        self.pos = nn.Linear(128, len(POSITIONS))

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.shape(h), self.color(h), self.pos(h)


def to_x(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images).float().permute(0, 3, 1, 2) / 127.5 - 1.0


def labels_to_t(labels: np.ndarray):
    return (torch.from_numpy(labels[:, 0]), torch.from_numpy(labels[:, 1]),
            torch.from_numpy(labels[:, 2]))


clf = Cls().to(device)
opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
train_x = to_x(train_images)
train_t = labels_to_t(train_labels)
for epoch in range(5):
    perm = torch.randperm(len(train_x))
    total = 0.0
    for start in range(0, len(train_x), 128):
        idx = perm[start:start + 128]
        x = train_x[idx].to(device)
        y = [t[idx].to(device) for t in train_t]
        opt.zero_grad(set_to_none=True)
        outs = clf(x)
        loss = sum(F.cross_entropy(o, t) for o, t in zip(outs, y))
        loss.backward()
        opt.step()
        total += float(loss) * len(idx)
    print(f"classifier epoch {epoch} loss {total / len(train_x):.4f}", flush=True)

clf.eval()


@torch.no_grad()
def acc(images: np.ndarray, labels: np.ndarray):
    x = to_x(images)
    outs = [[], [], []]
    for start in range(0, len(x), 256):
        o = clf(x[start:start + 256].to(device))
        for i in range(3):
            outs[i].append(o[i].argmax(-1).cpu())
    pred = np.stack([torch.cat(o).numpy() for o in outs], axis=1)
    return [float((pred[:, i] == labels[:, i]).mean()) for i in range(3)]


real = acc(test_images, test_labels)
fake = acc(gen_np, test_labels)
result = {
    "experiment": "KALIA-Vision-v0-eval",
    "ckpt_step": payload.get("step"),
    "real_test_shape": real[0],
    "real_test_color": real[1],
    "real_test_position": real[2],
    "generated_shape": fake[0],
    "generated_color": fake[1],
    "generated_position": fake[2],
    "W-1_shape": fake[0] >= 0.80,
    "W-2_color": fake[1] >= 0.80,
    "W-3_position": fake[2] >= 0.70,
    "n": len(gen_np),
    "grid": str(out_dir / "grid.png"),
}
out_dir / "eval.json".write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(result, indent=2, sort_keys=True), flush=True)
