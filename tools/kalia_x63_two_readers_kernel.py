"""Kaggle GPU kernel: X63 two-reader ensemble.

Sample N=8 candidates per target; score each with the CNN captioner and the
ViT captioner; select by max(CE_cnn, CE_vit) (both must agree). The
independent classifier judges and is never used for selection.
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
BATCH_TARGETS = 16
BASE_REPO = "kalia-lm/kalia-v031"
BASE_SHA = "bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468"
CODE_COMMIT = "8e1ee621f7c07569e494ad08d656e3e6180fa7a2"

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
             "vision_conditioning.py", "synthetic_vision_v2.py"):
    urllib.request.urlretrieve(f"{RAW}/{name}", work / name)
ckpt_dir = work / "ckpts"; ckpt_dir.mkdir(exist_ok=True)
CKPTS = {
    "ckpt64.pt": f"{RAW}/docs/results/kalia-vision-64-large/ckpt.pt",
    "cap64.pt": f"{RAW}/docs/results/kalia-vision-64/ckpt_captioner.pt",
    "clf64.pt": f"{RAW}/docs/results/kalia-vision-64-large/classifier.pt",
    "vit64.pt": f"{RAW}/docs/results/kalia-vision-64-vit/ckpt.pt",
}
for name, url in CKPTS.items():
    if not (ckpt_dir / name).exists():
        urllib.request.urlretrieve(url, ckpt_dir / name)
    print(name, (ckpt_dir / name).stat().st_size, flush=True)

import synthetic_vision_v2 as sv2  # noqa: E402
from vision_captioner import (ViTImageEncoder, build_batch, caption_images,
                              make_model)  # noqa: E402
from vision_conditioning import caption_embeddings  # noqa: E402
from vision_diffusion import Diffusion, UNetLarge  # noqa: E402
from edit_compiler import apply_lora, load_lora  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"
torch.set_float32_matmul_precision("high")
torch.manual_seed(63)

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

cap_cnn, enc_cnn, _ = make_model(base_ckpt, device=device)
payload = torch.load(ckpt_dir / "cap64.pt", map_location="cpu", weights_only=False)
load_lora(cap_cnn, ckpt_dir / "cap64.pt")
enc_cnn.load_state_dict(payload["encoder"])
cap_cnn.eval(); enc_cnn.eval()

cap_vit = GPT(GPTConfig(**cfg))
cap_vit.load_state_dict(base.state_dict(), strict=False)
cap_vit.to(device)
for p in cap_vit.parameters():
    p.requires_grad = False
apply_lora(cap_vit, 16, 16)
cap_vit.to(device)
enc_vit = ViTImageEncoder().to(device)
payload_vit = torch.load(ckpt_dir / "vit64.pt", map_location="cpu", weights_only=False)
cap_vit.load_state_dict(payload_vit["lora"], strict=False)
enc_vit.load_state_dict(payload_vit["encoder"])
cap_vit.eval(); enc_vit.eval()

unet = UNetLarge().to(device)
unet.load_state_dict(torch.load(ckpt_dir / "ckpt64.pt", map_location="cpu",
                                weights_only=False)["model"])
unet.eval()
diff = Diffusion(1000, device, image_size=64)


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


clf = Cls64().to(device)
clf.load_state_dict(torch.load(ckpt_dir / "clf64.pt", map_location="cpu",
                               weights_only=False)["model"])
clf.eval()


@torch.no_grad()
def verifier_scores(cap, enc_model, images, captions):
    imgs_t, x, y, mask, attn = build_batch(images, captions, enc, device)
    prefix = enc_model(imgs_t)
    logits, _ = cap(x, prefix_embeds=prefix)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                         ignore_index=-100, reduction="none").view(y.shape)
    return (ce * mask).sum(1) / mask.sum(1).clamp(min=1)


@torch.no_grad()
def clf_pred(images):
    outs = [[], [], [], [], [], []]
    for start in range(0, len(images), 128):
        o = clf(images[start:start + 128].to(device))
        for i in range(6):
            outs[i].append(o[i].argmax(-1).cpu())
    return torch.stack([torch.cat(o) for o in outs], dim=1).numpy()


print("targets", flush=True)
all_combos = sv2.all_combos()
_, held = sv2.split_combos()
held = set(tuple(c) for c in held)
imgs, recs = sv2.generate(5000, 33, all_combos)
idx = [i for i, r in enumerate(recs) if tuple(r["combo"]) in held][:TARGETS]
caps = [recs[i]["caption"] for i in idx]
labels = np.array([sv2.parse_combo(c) for c in caps], dtype=np.int64)
print("n", len(caps), flush=True)

single_ok = max_ok = mean_ok = oracle_ok = 0
selected = []
for start in range(0, len(caps), BATCH_TARGETS):
    b_caps = caps[start:start + BATCH_TARGETS]
    b_labels = labels[start:start + BATCH_TARGETS]
    b = len(b_caps)
    emb = caption_embeddings(base_text, enc, b_caps, mode="bidir", device=device).to(device)
    emb = emb.repeat_interleave(N, dim=0)
    gen = diff.sample(unet, emb, steps=100)
    cand_caps = [c for c in b_caps for _ in range(N)]
    s_cnn = verifier_scores(cap_cnn, enc_cnn, gen, cand_caps).view(b, N)
    s_vit = verifier_scores(cap_vit, enc_vit, gen, cand_caps).view(b, N)
    sel_single = s_cnn.argmin(dim=1).cpu().numpy()
    sel_max = torch.maximum(s_cnn, s_vit).argmin(dim=1).cpu().numpy()
    sel_mean = (0.5 * s_cnn + 0.5 * s_vit).argmin(dim=1).cpu().numpy()
    preds = clf_pred(gen).reshape(b, N, 6)
    exact = (preds == b_labels[:, None, :]).all(-1)
    ar = np.arange(b)
    single_ok += int(exact[ar, sel_single].sum())
    max_ok += int(exact[ar, sel_max].sum())
    mean_ok += int(exact[ar, sel_mean].sum())
    oracle_ok += int(exact.any(1).sum())
    gen5 = gen.reshape(b, N, 3, 64, 64)
    chosen = gen5[torch.arange(b, device=gen.device), torch.tensor(sel_max, device=gen.device)]
    selected.append(chosen.cpu())
    done = start + b
    print(done, "/", len(caps), "single", single_ok / done,
          "max", max_ok / done, "mean", mean_ok / done, flush=True)

n = len(caps)
single_exact = single_ok / n
max_exact = max_ok / n
mean_exact = mean_ok / n
oracle_exact = oracle_ok / n
print("single", single_exact, "max", max_exact, "mean", mean_exact,
      "oracle", oracle_exact, flush=True)

out = Path("/kaggle/working/two-readers"); out.mkdir(parents=True, exist_ok=True)
sel = torch.cat(selected, 0)
arr = ((sel.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
np.save(out / "selected_64.npy", arr)
canvas = Image.new("RGB", (8 * 128, 8 * 128), "white")
for i in range(min(64, len(arr))):
    r, c = divmod(i, 8)
    canvas.paste(Image.fromarray(arr[i]).resize((128, 128), Image.NEAREST),
                 (c * 128, r * 128))
canvas.save(out / "selected_64.png")

metrics = {
    "experiment": "X63-two-readers",
    "base_sha256": base_sha,
    "code_commit": CODE_COMMIT,
    "n": n, "N": N,
    "single_cnn_exact": single_exact,
    "ensemble_max_exact": max_exact,
    "ensemble_mean_exact": mean_exact,
    "oracle_exact": oracle_exact,
    "gain_max": max_exact - single_exact,
    "gain_mean": mean_exact - single_exact,
    "E-1": max_exact >= 0.97,
    "E-2": (max_exact - single_exact) >= 0.015,
    "E-3": True,
    "E-4": True,
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
