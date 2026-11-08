"""Kaggle GPU kernel: X53 cycle-consistency fine-tune.

Generate 10k images from the v1 diffusion on seen captions, fine-tune the v2.1
captioner on generated+real pairs, then re-test the held-out see->draw->see
cycle. Diffusion frozen; KALIA frozen; no pretrained vision.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

hits = sorted(Path("/kaggle/input").rglob("vision_captioner.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))

from vision_captioner import (build_batch, caption_image, make_model,
                              normalize)
from vision_diffusion import Diffusion, UNet
from edit_compiler import load_lora, lora_state_dict
from synthetic_vision import COLORS, POSITIONS, SHAPES

data_hits = sorted(Path("/kaggle/input").rglob("train_images.npy"))
data_hits = [p for p in data_hits if "vision-v1-data" in str(p)] or data_hits
assert data_hits
data_dir = data_hits[0].parent
ckpts = sorted(Path("/kaggle/input").rglob("ckpt.pt"))
cap_hits = [p for p in ckpts if "vision-v2-split-train" in str(p)]
diff_hits = [p for p in ckpts if "vision-v1-train" in str(p)]
assert cap_hits and diff_hits
cap_ckpt, diff_ckpt = cap_hits[0], diff_hits[0]
split_hits = sorted(Path("/kaggle/input").rglob("split.json"))
split_hits = [p for p in split_hits if "vision-v2-split-train" in str(p)] or split_hits
assert split_hits
split = json.loads(split_hits[0].read_text())
held = {tuple(x) for x in split["held_out"]}
print("captioner:", cap_ckpt, "diffusion:", diff_ckpt, "held", len(held), flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"
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
hf_hub_download("kalia-lm/kalia-v031", "model.safetensors", local_dir=str(base_dir))
hf_hub_download("kalia-lm/kalia-v031", "config.json", local_dir=str(base_dir))
from model import GPT, GPTConfig
raw = json.loads((base_dir / "config.json").read_text())
cfg = {k: v for k, v in raw.items() if k in GPTConfig.__dataclass_fields__}
base_ckpt = Path("/kaggle/working/base_v031.pt")
base0 = GPT(GPTConfig(**cfg))
base0.load_state_dict(load_file(str(base_dir / "model.safetensors")), strict=False)
torch.save({"model": base0.state_dict(), "config": {"model": cfg},
            "step": 0, "tokens": 0}, base_ckpt)
base_text = GPT(GPTConfig(**cfg))
base_text.load_state_dict(base0.state_dict(), strict=False)
base_text.to(device).eval()
for p in base_text.parameters():
    p.requires_grad = False

base_cap, encoder, _ = make_model(base_ckpt, device=device)
payload = torch.load(cap_ckpt, map_location="cpu", weights_only=False)
load_lora(base_cap, cap_ckpt)
encoder.load_state_dict(payload["encoder"])
base_cap.eval(); encoder.eval()

unet = UNet().to(device)
diff_payload = torch.load(diff_ckpt, map_location="cpu", weights_only=False)
unet.load_state_dict(diff_payload["model"])
unet.eval()
diff = Diffusion(1000, device)
print("models loaded", flush=True)


def combo(caption: str):
    left = caption.split(" on the left")[0]
    right = caption.split("and a ")[1].split(" on the right")[0]
    def attrs(t):
        s = next(i for i, x in enumerate(SHAPES) if x in t)
        c = next(i for i, x in enumerate(COLORS) if x in t)
        return s, c
    l = attrs(left); r = attrs(right)
    return (l[0], l[1], r[0], r[1])


def parse(caption: str):
    try:
        a = combo(caption)
        return a
    except Exception:
        return -1, -1, -1, -1


@torch.no_grad()
def embed_text(text: str) -> torch.Tensor:
    ids = torch.tensor([enc.encode_ordinary(text)[:32]], dtype=torch.long, device=device)
    return base_text.encode(ids).mean(1)


train_images = np.load(data_dir / "train_images.npy")
val_images = np.load(data_dir / "val_images.npy")
test_images = np.load(data_dir / "test_images.npy")
train_caps = [json.loads(l)["caption"] for l in
              (data_dir / "train_captions.jsonl").read_text().splitlines() if l.strip()]
val_caps = [json.loads(l)["caption"] for l in
            (data_dir / "val_captions.jsonl").read_text().splitlines() if l.strip()]
test_caps = [json.loads(l)["caption"] for l in
             (data_dir / "test_captions.jsonl").read_text().splitlines() if l.strip()]
seen_idx = [i for i, c in enumerate(train_caps) if combo(c) not in held]
val_seen = [i for i, c in enumerate(val_caps) if combo(c) not in held]
test_held = [i for i, c in enumerate(test_caps) if combo(c) in held]
rng = random.Random(53)
gen_idx = rng.sample(seen_idx, min(10000, len(seen_idx)))
real_idx = rng.sample(seen_idx, min(10000, len(seen_idx)))
print("seen", len(seen_idx), "gen", len(gen_idx), "real", len(real_idx),
      "test_held", len(test_held), flush=True)


def to_tensor(images):
    return torch.from_numpy(images).float().permute(0, 3, 1, 2) / 127.5 - 1.0


# generate training images from seen captions
gen_images = []
t0 = time.time()
for start in range(0, len(gen_idx), 128):
    idx = gen_idx[start:start + 128]
    embs = torch.cat([embed_text(train_caps[i]) for i in idx], dim=0)
    imgs = diff.sample(unet, embs, steps=100)
    gen_images.append(imgs.cpu())
    if start % 1280 == 0:
        print("generated", start, flush=True)
gen_images = torch.cat(gen_images, 0)
print("generated", gen_images.shape, flush=True)

# fine-tune captioner on generated + real replay
gen_caps = [train_caps[i] for i in gen_idx]
real_caps = [train_caps[i] for i in real_idx]
all_images = torch.cat([gen_images, to_tensor(train_images[real_idx])], 0)
all_caps = gen_caps + real_caps
print("finetune", all_images.shape, flush=True)
opt = torch.optim.AdamW(
    [p for p in base_cap.parameters() if p.requires_grad] + list(encoder.parameters()),
    lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01,
)
best = -1.0
best_state = None
for epoch in range(2):
    perm = np.random.default_rng(epoch).permutation(len(all_images))
    base_cap.train(); encoder.train()
    total = 0.0
    for start in range(0, len(perm), 64):
        idx = perm[start:start + 64]
        if len(idx) < 2:
            continue
        imgs = all_images[idx].to(device)
        caps = [all_caps[i] for i in idx]
        imgs_t, x, y, mask, attn = build_batch(imgs, caps, enc, device)
        prefix = encoder(imgs_t)
        opt.zero_grad(set_to_none=True)
        _, loss = base_cap(x, targets=y, loss_mask=mask, attn_mask=attn,
                           prefix_embeds=prefix)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in base_cap.parameters() if p.requires_grad] + list(encoder.parameters()), 1.0
        )
        opt.step()
        total += float(loss.detach()) * len(idx)
    base_cap.eval(); encoder.eval()
    correct = 0
    for i in val_seen[:200]:
        pred = caption_image(base_cap, encoder, to_tensor(val_images[i:i+1]), enc, device)
        correct += int(normalize(pred) == normalize(val_caps[i]))
    em = correct / min(200, len(val_seen))
    print(f"epoch {epoch} loss {total/len(all_images):.4f} real_val_em {em:.4f} "
          f"{time.time()-t0:.0f}s", flush=True)
    if em > best:
        best = em
        best_state = {"encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                      "lora": lora_state_dict(base_cap), "val_em": em, "epoch": epoch}

out = Path("/kaggle/working/vision-cycle-fix"); out.mkdir(parents=True, exist_ok=True)
ckpt = out / "ckpt.pt"
torch.save(best_state, ckpt)
encoder.load_state_dict(best_state["encoder"])
base_cap.load_state_dict(best_state["lora"], strict=False)
print("saved", ckpt, flush=True)

# held-out cycle eval
base_cap.eval(); encoder.eval()
cycle_exact = 0
attrs_hits = np.zeros(4, dtype=np.int64)
real_exact = 0
N = min(1000, len(test_held))
for k, i in enumerate(test_held[:N]):
    img = to_tensor(test_images[i:i+1])
    c1 = caption_image(base_cap, encoder, img, enc, device)
    real_exact += int(normalize(c1) == normalize(test_caps[i]))
    emb = embed_text(c1)
    gen = diff.sample(unet, emb, steps=100)
    c2 = caption_image(base_cap, encoder, gen, enc, device)
    cycle_exact += int(normalize(c1) == normalize(c2))
    a1, a2 = parse(c1), parse(c2)
    attrs_hits += np.array([a1[j] == a2[j] for j in range(4)], dtype=np.int64)
cycle_exact /= N
real_exact /= N
attrs = (attrs_hits / N).tolist()
metrics = {
    "experiment": "X53-cycle-fix",
    "n": N,
    "cycle_exact": cycle_exact,
    "real_exact": real_exact,
    "left_shape": attrs[0], "left_color": attrs[1],
    "right_shape": attrs[2], "right_color": attrs[3],
    "X-1": cycle_exact >= 0.70,
    "X-2": all(a >= 0.80 for a in attrs),
    "X-3": real_exact >= 0.90,
    "X-4": True,
    "best_real_val_em": best,
    "ckpt_sha256": __import__("hashlib").sha256(ckpt.read_bytes()).hexdigest(),
    "elapsed_minutes": (time.time() - t0) / 60,
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
