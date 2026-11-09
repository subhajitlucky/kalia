"""Kaggle GPU kernel: X60 self-verifying visual generation.

Sample N=8 candidates per target caption; score each with the frozen
captioner's teacher-forced CE of the target; select the minimum; judge with an
independent classifier. No external reward model, no human labels, no
distillation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

N = 8
TARGETS = 1000
B64 = 16
B32 = 16
BASE_REPO = "kalia-lm/kalia-v031"
BASE_SHA = "bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468"
CODE_COMMIT = "6072ca2975fbe09b551ef3c2d5094c1739e7c6a8"

hits = sorted(Path("/kaggle/input").rglob("vision_captioner.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))

RAW = f"https://raw.githubusercontent.com/subhajitlucky/kalia/{CODE_COMMIT}"
for name in ("vision_captioner.py", "vision_diffusion.py",
             "vision_conditioning.py", "synthetic_vision_v1.py",
             "synthetic_vision_v2.py"):
    urllib.request.urlretrieve(f"{RAW}/{name}", work / name)
urllib.request.urlretrieve(
    f"{RAW}/docs/results/kalia-vision-v2.1/split.json", work / "split32.json")

ckpt_dir = work / "ckpts"
ckpt_dir.mkdir(exist_ok=True)
CKPTS = {
    "ckpt32.pt": f"{RAW}/docs/results/kalia-vision-v1/ckpt.pt",
    "cap32.pt": f"{RAW}/docs/results/kalia-vision-v2.1/ckpt.pt",
    "ckpt64.pt": f"{RAW}/docs/results/kalia-vision-64-large/ckpt.pt",
    "cap64.pt": f"{RAW}/docs/results/kalia-vision-64/ckpt_captioner.pt",
    "clf64.pt": f"{RAW}/docs/results/kalia-vision-64-large/classifier.pt",
}
for name, url in CKPTS.items():
    if not (ckpt_dir / name).exists():
        urllib.request.urlretrieve(url, ckpt_dir / name)
    print(name, (ckpt_dir / name).stat().st_size, flush=True)

import synthetic_vision_v1 as sv1  # noqa: E402
import synthetic_vision_v2 as sv2  # noqa: E402
from vision_captioner import (build_batch, caption_images, make_model,
                              normalize)  # noqa: E402
from vision_conditioning import caption_embeddings  # noqa: E402
from vision_diffusion import Diffusion, UNet, UNetLarge  # noqa: E402
from edit_compiler import load_lora  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"
torch.set_float32_matmul_precision("high")
torch.manual_seed(60)

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
from model import GPT, GPTConfig  # noqa: E402
cfg = {k: v for k, v in raw_cfg.items() if k in GPTConfig.__dataclass_fields__}
base = GPT(GPTConfig(**cfg))
load = base.load_state_dict(load_file(sf), strict=False)
assert not load.missing_keys and not load.unexpected_keys
base_ckpt = Path("/kaggle/working/base_v031.pt")
torch.save({"model": base.state_dict(), "config": {"model": cfg},
            "step": 0, "tokens": 0}, base_ckpt)
base_text = GPT(GPTConfig(**cfg))
base_text.load_state_dict(base.state_dict(), strict=False)
base_text.to(device).eval()
for p in base_text.parameters():
    p.requires_grad = False


def load_captioner(path):
    m, e, _ = make_model(base_ckpt, device=device)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    load_lora(m, path)
    e.load_state_dict(payload["encoder"])
    m.eval(); e.eval()
    return m, e


cap32, enc32 = load_captioner(ckpt_dir / "cap32.pt")
cap64, enc64 = load_captioner(ckpt_dir / "cap64.pt")

unet32 = UNet().to(device)
unet32.load_state_dict(torch.load(ckpt_dir / "ckpt32.pt", map_location="cpu",
                                  weights_only=False)["model"])
unet32.eval()
diff32 = Diffusion(1000, device, image_size=32)

unet64 = UNetLarge().to(device)
unet64.load_state_dict(torch.load(ckpt_dir / "ckpt64.pt", map_location="cpu",
                                  weights_only=False)["model"])
unet64.eval()
diff64 = Diffusion(1000, device, image_size=64)
print("models loaded", flush=True)


class Cls64(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GroupNorm(8, 256), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.ls = nn.Linear(256, len(sv2.SHAPES))
        self.lc = nn.Linear(256, len(sv2.COLOR_NAMES))
        self.cs = nn.Linear(256, len(sv2.SHAPES))
        self.cc = nn.Linear(256, len(sv2.COLOR_NAMES))
        self.rs = nn.Linear(256, len(sv2.SHAPES))
        self.rc = nn.Linear(256, len(sv2.COLOR_NAMES))

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.ls(h), self.lc(h), self.cs(h), self.cc(h), self.rs(h), self.rc(h)


class Cls32(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.ls = nn.Linear(128, len(sv1.SHAPES))
        self.lc = nn.Linear(128, len(sv1.COLORS))
        self.rs = nn.Linear(128, len(sv1.SHAPES))
        self.rc = nn.Linear(128, len(sv1.COLORS))

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.ls(h), self.lc(h), self.rs(h), self.rc(h)


clf64 = Cls64().to(device)
clf64.load_state_dict(torch.load(ckpt_dir / "clf64.pt", map_location="cpu",
                                 weights_only=False)["model"])
clf64.eval()


def to_tensor(images):
    return torch.from_numpy(images).float().permute(0, 3, 1, 2) / 127.5 - 1.0


SHAPES32 = ("circle", "square", "triangle")
COLORS32 = ("red", "green", "blue", "yellow")


def parse32(caption: str):
    left = caption.split(" on the left")[0]
    right = caption.split("and a ")[1].split(" on the right")[0]

    def attrs(t):
        s = next(i for i, x in enumerate(SHAPES32) if x in t)
        c = next(i for i, x in enumerate(COLORS32) if x in t)
        return s, c

    l = attrs(left); r = attrs(right)
    return (l[0], l[1], r[0], r[1])


print("building 32x32 classifier", flush=True)
tr_images, tr_recs = sv1.generate_split(20000, 21)
tr_labels = np.array([parse32(r["caption"]) for r in tr_recs], dtype=np.int64)
clf32 = Cls32().to(device)
opt = torch.optim.AdamW(clf32.parameters(), lr=1e-3)
x32 = to_tensor(tr_images)
y32 = tuple(torch.from_numpy(tr_labels[:, i]) for i in range(4))
for epoch in range(5):
    perm = torch.randperm(len(x32))
    total = 0.0
    for start in range(0, len(x32), 128):
        idx = perm[start:start + 128]
        xb = x32[idx].to(device)
        yb = [t[idx].to(device) for t in y32]
        opt.zero_grad(set_to_none=True)
        outs = clf32(xb)
        loss = sum(F.cross_entropy(o, t) for o, t in zip(outs, yb))
        loss.backward()
        opt.step()
        total += float(loss.detach()) * len(idx)
    print("clf32 epoch", epoch, total / len(x32), flush=True)
clf32.eval()


@torch.no_grad()
def verifier_scores(cap, enc_model, images, captions, enc, device):
    imgs_t, x, y, mask, attn = build_batch(images, captions, enc, device)
    prefix = enc_model(imgs_t)
    logits, _ = cap(x, prefix_embeds=prefix)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                         ignore_index=-100, reduction="none").view(y.shape)
    return (ce * mask).sum(1) / mask.sum(1).clamp(min=1)


@torch.no_grad()
def clf_pred(clf, images, device, heads):
    x = images if isinstance(images, torch.Tensor) else to_tensor(images)
    outs = [[] for _ in range(heads)]
    for start in range(0, len(x), 256):
        o = clf(x[start:start + 256].to(device))
        for i in range(heads):
            outs[i].append(o[i].argmax(-1).cpu())
    return torch.stack([torch.cat(o) for o in outs], dim=1).numpy()


def grid_images(images, path, n=64, cell=128):
    imgs = images[:n]
    cols = 8
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), "white")
    for i, im in enumerate(imgs):
        arr = ((im.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr).resize((cell, cell), Image.NEAREST)
        r, c = divmod(i, cols)
        canvas.paste(pil, (c * cell, r * cell))
    canvas.save(path)


@torch.no_grad()
def run_dataset(name, caps, labels, unet, diff, cap, enc_model, clf, heads,
                image_size, batch_targets):
    n = len(caps)
    n1_ok = sel_ok = oracle_ok = ver_ok = 0
    selected_all = []
    for start in range(0, n, batch_targets):
        b_caps = caps[start:start + batch_targets]
        b_labels = labels[start:start + batch_targets]
        b = len(b_caps)
        emb = caption_embeddings(base_text, enc, b_caps, mode="bidir", device=device).to(device)
        emb = emb.repeat_interleave(N, dim=0)
        gen = diff.sample(unet, emb, steps=100)
        cand_caps = [c for c in b_caps for _ in range(N)]
        scores = verifier_scores(cap, enc_model, gen, cand_caps, enc, device).view(b, N)
        idx = scores.argmin(dim=1)
        gen5 = gen.reshape(b, N, 3, image_size, image_size)
        sel = gen5[torch.arange(b, device=gen.device), idx]
        preds = clf_pred(clf, gen, device, heads).reshape(b, N, heads)
        exact = (preds == b_labels[:, None, :]).all(-1)
        sel_idx = idx.cpu().numpy()
        n1_ok += int(exact[:, 0].sum())
        sel_ok += int(exact[np.arange(b), sel_idx].sum())
        oracle_ok += int(exact.any(1).sum())
        read = caption_images(cap, enc_model, sel, enc, device)
        ver_ok += sum(normalize(r) == normalize(c) for r, c in zip(read, b_caps))
        selected_all.append(sel.cpu())
        print(name, start + b, "/", n, "sel", sel_ok / (start + b),
              "n1", n1_ok / (start + b), flush=True)
    selected = torch.cat(selected_all, 0)
    return {
        "n": n, "n1_exact": n1_ok / n, "selected_exact": sel_ok / n,
        "oracle_exact": oracle_ok / n, "verifier_exact": ver_ok / n,
        "gain": (sel_ok - n1_ok) / n,
    }, selected


print("64x64 targets", flush=True)
all64 = sv2.all_combos()
_, held64 = sv2.split_combos()
held64 = set(tuple(c) for c in held64)
imgs64, recs64 = sv2.generate(5000, 33, all64)
idx64 = [i for i, r in enumerate(recs64) if tuple(r["combo"]) in held64][:TARGETS]
caps64 = [recs64[i]["caption"] for i in idx64]
labels64 = np.array([sv2.parse_combo(c) for c in caps64], dtype=np.int64)
print("64 targets", len(caps64), flush=True)
res64, sel64 = run_dataset("64", caps64, labels64, unet64, diff64, cap64, enc64,
                           clf64, 6, 64, B64)
print("64 result", res64, flush=True)

print("32x32 targets", flush=True)
split32 = json.loads((work / "split32.json").read_text())
held32 = set(tuple(x) for x in split32["held_out"])
imgs32, recs32 = sv1.generate_split(5000, 23)
idx32 = [i for i, r in enumerate(recs32) if parse32(r["caption"]) in held32][:TARGETS]
caps32 = [recs32[i]["caption"] for i in idx32]
labels32 = np.array([parse32(c) for c in caps32], dtype=np.int64)
print("32 targets", len(caps32), flush=True)
res32, sel32 = run_dataset("32", caps32, labels32, unet32, diff32, cap32, enc32,
                           clf32, 4, 32, B32)
print("32 result", res32, flush=True)

out = Path("/kaggle/working/selfverify"); out.mkdir(parents=True, exist_ok=True)
grid_images(sel64, out / "selected_64.png")
grid_images(sel32, out / "selected_32.png")
np.save(out / "selected_64.npy",
        ((sel64.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8))
np.save(out / "selected_32.npy",
        ((sel32.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8))

metrics = {
    "experiment": "X60-selfverify",
    "base_sha256": base_sha,
    "code_commit": CODE_COMMIT,
    "N": N,
    "64": res64,
    "32": res32,
    "S-1": res64["selected_exact"] >= 0.90,
    "S-2": res32["selected_exact"] >= 0.98,
    "S-3": (res64["gain"] >= 0.10) and (res32["gain"] >= 0.03),
    "S-4": True,
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
