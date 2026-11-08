"""Kaggle GPU kernel: X58 64x64 three-object captioner.

From-scratch CNN encoder + 4 image tokens + rank-16 LoRA on frozen KALIA
v0.3.1. Trains on seen combinations, tests on a held-out 20% of the 13,824
three-object combinations.
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
import torch.nn.functional as F

EPOCHS = 8
BATCH = 64
LR = 2e-4
VAL_N = 200
TEST_N = 1000
BASE_REPO = "kalia-lm/kalia-v031"
BASE_SHA = "bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468"
CODE_COMMIT = "216e21db9c477431fe191239ac51e9a65c572285"

hits = sorted(Path("/kaggle/input").rglob("vision_captioner.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))

for name in ("synthetic_vision_v2.py", "vision_captioner.py"):
    path = work / name
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/subhajitlucky/kalia/{CODE_COMMIT}/{name}",
        path,
    )
import synthetic_vision_v2 as sv  # noqa: E402
from vision_captioner import (build_batch, caption_images, make_model,
                              normalize)  # noqa: E402
from edit_compiler import lora_state_dict  # noqa: E402

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
from model import GPT, GPTConfig  # noqa: E402
cfg = {k: v for k, v in raw_cfg.items() if k in GPTConfig.__dataclass_fields__}
base = GPT(GPTConfig(**cfg))
load = base.load_state_dict(load_file(sf), strict=False)
assert not load.missing_keys and not load.unexpected_keys
base_ckpt = Path("/kaggle/working/base_v031.pt")
torch.save({"model": base.state_dict(), "config": {"model": cfg},
            "step": 0, "tokens": 0}, base_ckpt)
print("base:", base_sha[:16], flush=True)

print("generating data", flush=True)
t0 = time.time()
seen, held = sv.split_combos()
train_images, train_recs = sv.generate(50000, 21, seen)
val_images, val_recs = sv.generate(5000, 22, seen)
test_images, test_recs = sv.generate(5000, 23, held)
seen_test_images, seen_test_recs = sv.generate(5000, 24, seen)
train_caps = [r["caption"] for r in train_recs]
val_caps = [r["caption"] for r in val_recs]
test_caps = [r["caption"] for r in test_recs]
seen_test_caps = [r["caption"] for r in seen_test_recs]
print("data", train_images.shape, val_images.shape, test_images.shape,
      f"{time.time()-t0:.0f}s", flush=True)


def to_tensor(images):
    return torch.from_numpy(images).float().permute(0, 3, 1, 2) / 127.5 - 1.0


def parse(caption: str):
    try:
        return sv.parse_combo(caption)
    except Exception:
        return (-1,) * 6


model, encoder, trainable = make_model(base_ckpt, device=device)
print("trainable params", trainable + sum(p.numel() for p in encoder.parameters()),
      flush=True)
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
    model.eval(); encoder.eval()
    correct = 0
    for i in range(min(VAL_N, len(val_images))):
        pred = caption_images(model, encoder, to_tensor(val_images[i:i+1]), enc, device)[0]
        correct += int(normalize(pred) == normalize(val_caps[i]))
    em = correct / min(VAL_N, len(val_images))
    print(f"epoch {epoch} loss {total/len(train_images):.4f} val_em {em:.4f} "
          f"{time.time()-t0:.0f}s", flush=True)
    if em > best:
        best = em
        best_state = {"encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                      "lora": lora_state_dict(model), "val_em": em, "epoch": epoch}

out = Path("/kaggle/working/vision64-captioner"); out.mkdir(parents=True, exist_ok=True)
ckpt = out / "ckpt.pt"
torch.save(best_state, ckpt)
encoder.load_state_dict(best_state["encoder"])
model.load_state_dict(best_state["lora"], strict=False)
model.eval(); encoder.eval()


def run_eval(images, caps, n):
    n = min(n, len(images))
    preds = []
    for start in range(0, n, 64):
        preds += caption_images(model, encoder, to_tensor(images[start:start+64]), enc, device)
    exact = sum(normalize(p) == normalize(c) for p, c in zip(preds, caps[:n])) / n
    labels = np.array([parse(c) for c in caps[:n]], dtype=np.int64)
    pred_lab = np.array([parse(p) for p in preds], dtype=np.int64)
    attrs = [float((pred_lab[:, i] == labels[:, i]).mean()) for i in range(6)]
    return exact, attrs, preds


held_exact, held_attrs, held_preds = run_eval(test_images, test_caps, TEST_N)
seen_exact, seen_attrs, _ = run_eval(seen_test_images, seen_test_caps, TEST_N)
print("held", held_exact, held_attrs, "seen", seen_exact, seen_attrs, flush=True)

# blind and shuffled controls on 200 held-out
blind, shuf = [], []
for i in range(200):
    img = to_tensor(test_images[i:i+1])
    idx = torch.tensor([enc.encode_ordinary("Describe: ")], dtype=torch.long, device=device)
    out_tok = []
    for _ in range(64):
        logits, _ = model(idx)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        tok = int(nxt.item())
        if tok in (198, 50256):
            break
        out_tok.append(tok); idx = torch.cat([idx, nxt], 1)
    blind.append(normalize(enc.decode(out_tok)))
    wrong = to_tensor(test_images[(i + 1) % len(test_images):(i + 1) % len(test_images) + 1])
    shuf.append(caption_images(model, encoder, wrong, enc, device)[0])
blind_exact = sum(normalize(p) == normalize(c) for p, c in zip(blind, test_caps[:200])) / 200
shuf_exact = sum(normalize(p) == normalize(c) for p, c in zip(shuf, test_caps[:200])) / 200

meta = {
    "experiment": "X58-vision-64-captioner",
    "base_sha256": base_sha,
    "code_commit": CODE_COMMIT,
    "trainable_params": trainable + sum(p.numel() for p in encoder.parameters()),
    "train_n": len(train_images), "val_n": len(val_images), "test_n": len(test_images),
    "held_combos": len(held), "seen_combos": len(seen),
    "best_val_em": best, "best_epoch": best_state["epoch"],
    "held_exact": held_exact, "seen_exact": seen_exact,
    "held_attrs": held_attrs, "seen_attrs": seen_attrs,
    "blind_exact": blind_exact, "shuffled_exact": shuf_exact,
    "ckpt_bytes": ckpt.stat().st_size,
    "ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    "V64-1": held_exact >= 0.90,
    "V64-2": seen_exact >= 0.95,
    "elapsed_minutes": (time.time() - t0) / 60,
}
(out / "metrics.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
(out / "split.json").write_text(json.dumps(
    {"held": [list(c) for c in held], "seen": len(seen), "held_n": len(held)},
    indent=2) + "\n")
print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
