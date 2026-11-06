"""Kaggle GPU kernel: train KALIA-Vision v0 conditional DDPM.

KALIA v0.3.1 is frozen and used only to precompute caption embeddings. The
32x32 diffusion U-Net is trained from scratch on the synthetic shapes data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MAX_STEPS = 20000
MAX_MINUTES = 480
BATCH = 128
LR = 2e-4
MAX_TEXT_LEN = 32

BASE_REPO = "kalia-lm/kalia-v031"
BASE_SHA = "bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468"

hits = sorted(Path("/kaggle/input").rglob("vision_diffusion.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))

data_hits = sorted(Path("/kaggle/input").rglob("train_images.npy"))
assert data_hits, "kalia-vision-data not attached"
data_dir = data_hits[0].parent
print("data:", data_dir, flush=True)

try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "safetensors", "huggingface_hub", "tiktoken"], check=True)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import GPT, GPTConfig
from vision_diffusion import Diffusion, UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "this kernel needs a GPU"
print("device:", torch.cuda.get_device_name(0), flush=True)

base_dir = Path("/kaggle/working/base")
base_dir.mkdir(parents=True, exist_ok=True)
sf = hf_hub_download(BASE_REPO, "model.safetensors", local_dir=str(base_dir))
cf = hf_hub_download(BASE_REPO, "config.json", local_dir=str(base_dir))
base_sha = hashlib.sha256(Path(sf).read_bytes()).hexdigest()
assert base_sha == BASE_SHA, (base_sha, BASE_SHA)
raw_cfg = json.loads(Path(cf).read_text())
cfg = {k: v for k, v in raw_cfg.items() if k in GPTConfig.__dataclass_fields__}
text_model = GPT(GPTConfig(**cfg))
load = text_model.load_state_dict(load_file(sf), strict=False)
assert not load.missing_keys and not load.unexpected_keys
text_model.to(device).eval()
for p in text_model.parameters():
    p.requires_grad = False
print("KALIA text encoder:", base_sha[:16], flush=True)

try:
    import tiktoken
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"],
                   check=True)
    import tiktoken

enc = tiktoken.get_encoding("gpt2")


def load_captions(split: str) -> list[str]:
    path = data_dir / f"{split}_captions.jsonl"
    return [json.loads(line)["caption"] for line in path.read_text().splitlines() if line.strip()]


@torch.no_grad()
def embed_captions(captions: list[str]) -> np.ndarray:
    out = []
    for start in range(0, len(captions), 64):
        batch = captions[start:start + 64]
        ids = [enc.encode_ordinary(c)[:MAX_TEXT_LEN] for c in batch]
        width = max(len(x) for x in ids)
        x = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), width), dtype=torch.bool, device=device)
        for i, seq in enumerate(ids):
            x[i, :len(seq)] = torch.tensor(seq, device=device)
            mask[i, :len(seq)] = True
        h = text_model.encode(x, attn_mask=mask[:, None, None, :])
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


out_dir = Path("/kaggle/working/vision")
out_dir.mkdir(parents=True, exist_ok=True)
text_path = out_dir / "text_emb.npz"
if text_path.exists():
    z = np.load(text_path)
    train_emb, val_emb, test_emb = z["train"], z["val"], z["test"]
else:
    print("precomputing caption embeddings", flush=True)
    train_emb = embed_captions(load_captions("train"))
    val_emb = embed_captions(load_captions("val"))
    test_emb = embed_captions(load_captions("test"))
    np.savez(text_path, train=train_emb, val=val_emb, test=test_emb)
print("text embeddings:", train_emb.shape, val_emb.shape, test_emb.shape, flush=True)

train_images = np.load(data_dir / "train_images.npy")
val_images = np.load(data_dir / "val_images.npy")
test_images = np.load(data_dir / "test_images.npy")


def to_tensor(images: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 127.5 - 1.0
    return x


train_x = to_tensor(train_images)
val_x = to_tensor(val_images)
train_t = torch.from_numpy(train_emb).float()
val_t = torch.from_numpy(val_emb).float()
ds = TensorDataset(train_x, train_t)
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)

model = UNet().to(device)
diff = Diffusion(1000, device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99),
                        weight_decay=0.0)
print("UNet params:", sum(p.numel() for p in model.parameters()), flush=True)

step = 0
t0 = time.time()
model.train()
while step < MAX_STEPS:
    for x, text in loader:
        if step >= MAX_STEPS or (MAX_MINUTES and (time.time() - t0) / 60 >= MAX_MINUTES):
            break
        x = x.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = diff.loss(model, x, text)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step}/{MAX_STEPS} loss {loss.item():.4f} "
                  f"{(time.time() - t0):.0f}s", flush=True)
        step += 1
    if step >= MAX_STEPS or (MAX_MINUTES and (time.time() - t0) / 60 >= MAX_MINUTES):
        break

ckpt = out_dir / "ckpt.pt"
torch.save({
    "model": model.state_dict(),
    "step": step,
    "config": {"image_size": 32, "cond_dim": 128, "text_dim": 512},
    "base_sha256": base_sha,
}, ckpt)
meta = {
    "experiment": "KALIA-Vision-v0",
    "step": step,
    "unet_params": sum(p.numel() for p in model.parameters()),
    "ckpt_bytes": ckpt.stat().st_size,
    "ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    "base_sha256": base_sha,
    "data_dir": str(data_dir),
    "elapsed_minutes": (time.time() - t0) / 60,
}
(out_dir / "train_manifest.json").write_text(
    json.dumps(meta, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
