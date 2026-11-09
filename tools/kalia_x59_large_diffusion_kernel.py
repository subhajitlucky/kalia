"""Kaggle GPU kernel: X59 64x64 three-object large diffusion.

Trains the v1 U-Net architecture from scratch at 64x64 on the 13,824-combo
three-object grid. Frozen KALIA v0.3.1 supplies the bidirectional caption
embeddings. No pretrained vision weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

MAX_STEPS = 30000
BATCH = 64
LR = 2e-4
MAX_TEXT_LEN = 64
BASE_REPO = "kalia-lm/kalia-v031"
BASE_SHA = "bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468"
CODE_COMMIT = "21a3563b9c0645b663a02104f35ee567e0f62b07"

hits = sorted(Path("/kaggle/input").rglob("model.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))

for name in ("synthetic_vision_v2.py", "vision_diffusion.py",
             "vision_conditioning.py"):
    path = work / name
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/subhajitlucky/kalia/{CODE_COMMIT}/{name}",
        path,
    )
import synthetic_vision_v2 as sv  # noqa: E402
from vision_conditioning import caption_embeddings  # noqa: E402
from vision_diffusion import Diffusion, UNetLarge  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"
torch.set_float32_matmul_precision("high")
torch.manual_seed(58)

try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "safetensors", "huggingface_hub", "tiktoken"], check=True)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
import tiktoken
enc = tiktoken.get_encoding("gpt2")

base_dir = Path("/kaggle/working/base"); base_dir.mkdir(parents=True, exist_ok=True)
sf = hf_hub_download(BASE_REPO, "model.safetensors", local_dir=str(base_dir))
cf = hf_hub_download(BASE_REPO, "config.json", local_dir=str(base_dir))
base_sha = hashlib.sha256(Path(sf).read_bytes()).hexdigest()
assert base_sha == BASE_SHA, (base_sha, BASE_SHA)
raw_cfg = json.loads(Path(cf).read_text())
cfg = {k: v for k, v in raw_cfg.items() if k in GPTConfig.__dataclass_fields__}
base_text = GPT(GPTConfig(**cfg))
load = base_text.load_state_dict(load_file(sf), strict=False)
assert not load.missing_keys and not load.unexpected_keys
base_text.to(device).eval()
for p in base_text.parameters():
    p.requires_grad = False
print("base:", base_sha[:16], flush=True)

print("generating data", flush=True)
t0 = time.time()
combos = sv.all_combos()
train_images, train_recs = sv.generate(50000, 31, combos)
val_images, val_recs = sv.generate(5000, 32, combos)
print("data", train_images.shape, val_images.shape, f"{time.time()-t0:.0f}s",
      flush=True)


def to_tensor(images):
    return torch.from_numpy(images).float().permute(0, 3, 1, 2) / 127.5 - 1.0


train_caps = [r["caption"] for r in train_recs]
val_caps = [r["caption"] for r in val_recs]
print("embedding captions", flush=True)
t0 = time.time()
train_emb = caption_embeddings(base_text, enc, train_caps, mode="bidir",
                               device=device, batch=64)
val_emb = caption_embeddings(base_text, enc, val_caps, mode="bidir",
                             device=device, batch=64)
print("embeddings", train_emb.shape, f"{time.time()-t0:.0f}s", flush=True)

train_x = to_tensor(train_images)
val_x = to_tensor(val_images)
ds = TensorDataset(train_x, train_emb)
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True,
                    num_workers=0)

model = UNetLarge().to(device)
diff = Diffusion(1000, device, image_size=64)
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99),
                        weight_decay=0.0)
print("UNet params:", sum(p.numel() for p in model.parameters()), flush=True)

step = 0
t0 = time.time()
model.train()
while step < MAX_STEPS:
    for x, text in loader:
        if step >= MAX_STEPS:
            break
        x = x.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = diff.loss(model, x, text)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step}/{MAX_STEPS} loss {loss.item():.4f} "
                  f"{(time.time()-t0):.0f}s", flush=True)
        step += 1

out = Path("/kaggle/working/vision64-large"); out.mkdir(parents=True, exist_ok=True)
ckpt = out / "ckpt.pt"
torch.save({
    "model": model.state_dict(),
    "step": step,
    "config": {"image_size": 64, "cond_dim": 128, "text_dim": 512, "arch": "UNetLarge", "base": 96, "attn": "8x8"},
    "base_sha256": base_sha,
    "code_commit": CODE_COMMIT,
}, ckpt)
print("saved", ckpt, flush=True)


# ---------------------------------------------------------------------------
# 6-head classifier on real 64x64 scenes
# ---------------------------------------------------------------------------
class Cls(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GroupNorm(8, 256), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.ls = nn.Linear(256, len(sv.SHAPES))
        self.lc = nn.Linear(256, len(sv.COLOR_NAMES))
        self.cs = nn.Linear(256, len(sv.SHAPES))
        self.cc = nn.Linear(256, len(sv.COLOR_NAMES))
        self.rs = nn.Linear(256, len(sv.SHAPES))
        self.rc = nn.Linear(256, len(sv.COLOR_NAMES))

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.ls(h), self.lc(h), self.cs(h), self.cc(h), self.rs(h), self.rc(h)


labels = np.array([sv.parse_combo(r["caption"]) for r in train_recs], dtype=np.int64)
val_labels = np.array([sv.parse_combo(r["caption"]) for r in val_recs], dtype=np.int64)
y = tuple(torch.from_numpy(labels[:, i]) for i in range(6))
clf = Cls().to(device)
opt_c = torch.optim.AdamW(clf.parameters(), lr=1e-3)
for epoch in range(5):
    perm = torch.randperm(len(train_x))
    total = 0.0
    for start in range(0, len(train_x), 128):
        idx = perm[start:start + 128]
        xb = train_x[idx].to(device)
        yb = [t[idx].to(device) for t in y]
        opt_c.zero_grad(set_to_none=True)
        outs = clf(xb)
        loss = sum(F.cross_entropy(o, t) for o, t in zip(outs, yb))
        loss.backward()
        opt_c.step()
        total += float(loss.detach()) * len(idx)
    print(f"classifier epoch {epoch} loss {total/len(train_x):.4f}", flush=True)
clf.eval()


@torch.no_grad()
def clf_acc(images, labels_np):
    x = to_tensor(images)
    outs = [[] for _ in range(6)]
    for start in range(0, len(x), 256):
        o = clf(x[start:start + 256].to(device))
        for i in range(6):
            outs[i].append(o[i].argmax(-1).cpu())
    pred = np.stack([torch.cat(o).numpy() for o in outs], axis=1)
    per = [float((pred[:, i] == labels_np[:, i]).mean()) for i in range(6)]
    exact = float(((pred == labels_np).all(axis=1)).mean())
    return per, exact


test_images, test_recs = sv.generate(5000, 33, combos)
test_caps = [r["caption"] for r in test_recs]
test_labels = np.array([sv.parse_combo(c) for c in test_caps], dtype=np.int64)
test_emb = caption_embeddings(base_text, enc, test_caps, mode="bidir",
                              device=device, batch=64)
real_per, real_exact = clf_acc(test_images, test_labels)
print("classifier real", real_per, real_exact, flush=True)

gen = []
with torch.no_grad():
    for start in range(0, len(test_emb), 128):
        emb = test_emb[start:start + 128].to(device)
        gen.append(diff.sample(model, emb, steps=100).cpu())
generated = torch.cat(gen, 0)
gen_np = ((generated.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
np.save(out / "generated_test.npy", gen_np)
from PIL import Image
grid = gen_np[:64]
canvas = np.zeros((8 * 64, 8 * 64, 3), dtype=np.uint8)
for i in range(64):
    r, c = divmod(i, 8)
    canvas[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = grid[i]
Image.fromarray(canvas).resize((512, 512), Image.NEAREST).save(out / "grid.png")
fake_per, fake_exact = clf_acc(gen_np, test_labels)
print("classifier generated", fake_per, fake_exact, flush=True)

torch.save({"model": clf.state_dict()}, out / "classifier.pt")
meta = {
    "experiment": "X59-vision-64-large-diffusion",
    "base_sha256": base_sha,
    "code_commit": CODE_COMMIT,
    "unet_params": sum(p.numel() for p in model.parameters()),
    "classifier_params": sum(p.numel() for p in clf.parameters()),
    "steps": step,
    "batch": BATCH,
    "ckpt_bytes": ckpt.stat().st_size,
    "ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    "real_exact": real_exact,
    "generated_exact": fake_exact,
    "real_attrs": real_per,
    "generated_attrs": fake_per,
    "X59-1": fake_exact >= 0.80,
    "elapsed_minutes": (time.time() - t0) / 60,
}
(out / "train_manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
