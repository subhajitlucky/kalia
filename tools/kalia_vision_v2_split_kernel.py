"""Kaggle GPU kernel: train KALIA-Vision v2 image encoder + caption decoder."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EPOCHS = 8
BATCH = 64
LR = 2e-4
VAL_N = 200
HELD_OUT_FRAC = 0.20
SPLIT_SEED = 2026
BASE_REPO = "kalia-lm/kalia-v031"
BASE_SHA = "bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468"

hits = sorted(Path("/kaggle/input").rglob("vision_captioner.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))

data_hits = sorted(Path("/kaggle/input").rglob("train_images.npy"))
data_hits = [p for p in data_hits if "vision-v1-data" in str(p)] or data_hits
assert data_hits, "vision v1 data not attached"
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

from model import GPT, GPTConfig
from vision_captioner import (PREFIX_TOKENS, build_batch, caption_image,
                              make_model, normalize)
from synthetic_vision import COLORS, SHAPES
from edit_compiler import lora_state_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"
torch.set_float32_matmul_precision("high")

base_dir = Path("/kaggle/working/base"); base_dir.mkdir(parents=True, exist_ok=True)
sf = hf_hub_download(BASE_REPO, "model.safetensors", local_dir=str(base_dir))
cf = hf_hub_download(BASE_REPO, "config.json", local_dir=str(base_dir))
base_sha = hashlib.sha256(Path(sf).read_bytes()).hexdigest()
assert base_sha == BASE_SHA
raw = json.loads(Path(cf).read_text())
cfg = {k: v for k, v in raw.items() if k in GPTConfig.__dataclass_fields__}
base = GPT(GPTConfig(**cfg))
load = base.load_state_dict(load_file(sf), strict=False)
assert not load.missing_keys and not load.unexpected_keys
base_ckpt = Path("/kaggle/working/base_v031.pt")
torch.save({"model": base.state_dict(), "config": {"model": cfg},
            "step": 0, "tokens": 0}, base_ckpt)
print("base:", base_sha[:16], flush=True)

try:
    import tiktoken
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"], check=True)
    import tiktoken
enc = tiktoken.get_encoding("gpt2")


def to_tensor(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images).float().permute(0, 3, 1, 2) / 127.5 - 1.0


train_images = np.load(data_dir / "train_images.npy")
val_images = np.load(data_dir / "val_images.npy")
test_images = np.load(data_dir / "test_images.npy")
train_caps = [json.loads(l)["caption"] for l in
              (data_dir / "train_captions.jsonl").read_text().splitlines() if l.strip()]
val_caps = [json.loads(l)["caption"] for l in
            (data_dir / "val_captions.jsonl").read_text().splitlines() if l.strip()]
test_caps = [json.loads(l)["caption"] for l in
             (data_dir / "test_captions.jsonl").read_text().splitlines() if l.strip()]
print("sizes", len(train_images), len(val_images), len(test_images), flush=True)

# deterministic held-out combination split
def combo(caption: str):
    left = caption.split(" on the left")[0]
    right = caption.split("and a ")[1].split(" on the right")[0]
    def attrs(t):
        s = next(i for i, x in enumerate(SHAPES) if x in t)
        c = next(i for i, x in enumerate(COLORS) if x in t)
        return s, c
    l = attrs(left); r = attrs(right)
    return (l[0], l[1], r[0], r[1])

all_combos = [(s1, c1, s2, c2) for s1 in range(len(SHAPES))
              for c1 in range(len(COLORS)) for s2 in range(len(SHAPES))
              for c2 in range(len(COLORS))]
rng_split = __import__("random").Random(SPLIT_SEED)
rng_split.shuffle(all_combos)
n_hold = int(round(HELD_OUT_FRAC * len(all_combos)))
held = set(all_combos[:n_hold])
train_idx = [i for i, c in enumerate(train_caps) if combo(c) not in held]
val_idx = [i for i, c in enumerate(val_caps) if combo(c) not in held]
train_images = train_images[train_idx]; train_caps = [train_caps[i] for i in train_idx]
val_images = val_images[val_idx]; val_caps = [val_caps[i] for i in val_idx]
print("held-out combos", len(held), "train", len(train_images), "val", len(val_images), flush=True)

model, encoder, trainable = make_model(base_ckpt, device=device)
print("trainable params", trainable + sum(p.numel() for p in encoder.parameters()), flush=True)
opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad] + list(encoder.parameters()),
    lr=LR, betas=(0.9, 0.95), weight_decay=0.01,
)
t0 = time.time()
best = -1.0
best_state = None
for epoch in range(EPOCHS):
    perm = np.random.default_rng(epoch).permutation(len(train_images))
    model.train(); encoder.train()
    total = 0.0
    for start in range(0, len(perm), BATCH):
        idx = perm[start:start + BATCH]
        if len(idx) < 2:
            continue
        imgs = to_tensor(train_images[idx]).to(device)
        caps = [train_caps[i] for i in idx]
        imgs_t, x, y, mask, attn = build_batch(imgs, caps, enc, device)
        prefix = encoder(imgs_t)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y, loss_mask=mask, attn_mask=attn,
                        prefix_embeds=prefix)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] + list(encoder.parameters()), 1.0
        )
        opt.step()
        total += float(loss.detach()) * len(idx)
    # val exact match on a fixed subset
    model.eval(); encoder.eval()
    correct = 0
    for i in range(min(VAL_N, len(val_images))):
        pred = caption_image(model, encoder, to_tensor(val_images[i:i+1]), enc, device)
        if normalize(pred) == normalize(val_caps[i]):
            correct += 1
    em = correct / min(VAL_N, len(val_images))
    print(f"epoch {epoch} loss {total/len(train_images):.4f} val_em {em:.4f} "
          f"{time.time()-t0:.0f}s", flush=True)
    if em > best:
        best = em
        best_state = {"encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                      "lora": lora_state_dict(model), "val_em": em, "epoch": epoch}

out = Path("/kaggle/working/vision-v2-split"); out.mkdir(parents=True, exist_ok=True)
(out / "split.json").write_text(json.dumps({
    "held_out": [list(x) for x in sorted(held)],
    "n_held": len(held),
    "train_n": len(train_images),
    "n_held_out": len(held),
    "val_n": len(val_images),
    "seed": SPLIT_SEED,
}, indent=2, sort_keys=True) + "\n")
ckpt = out / "ckpt.pt"
torch.save(best_state, ckpt)
meta = {
    "experiment": "KALIA-Vision-v2.1-split",
    "base_sha256": base_sha,
    "train_n": len(train_images),
    "n_held_out": len(held),
    "val_n": len(val_images),
    "test_n": len(test_images),
    "best_val_em": best,
    "best_epoch": best_state["epoch"],
    "trainable_params": trainable + sum(p.numel() for p in encoder.parameters()),
    "ckpt_bytes": ckpt.stat().st_size,
    "ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    "elapsed_minutes": (time.time() - t0) / 60,
}
(out / "train_manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
