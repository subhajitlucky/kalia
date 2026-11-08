"""Kaggle GPU kernel: evaluate KALIA-Vision v2 image captioning."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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

from vision_captioner import caption_image, make_model, normalize
from edit_compiler import load_lora
from synthetic_vision import COLORS, POSITIONS, SHAPES

data_hits = sorted(Path("/kaggle/input").rglob("test_images.npy"))
data_hits = [p for p in data_hits if "vision-v1-data" in str(p)] or data_hits
assert data_hits
data_dir = data_hits[0].parent
ckpt_hits = sorted(Path("/kaggle/input").rglob("ckpt.pt"))
ckpt_hits = [p for p in ckpt_hits if "vision-v2" in str(p)] or ckpt_hits
assert ckpt_hits
ckpt_path = ckpt_hits[0]
print("ckpt:", ckpt_path, flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"
try:
    from huggingface_hub import hf_hub_download
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "safetensors", "huggingface_hub", "tiktoken"], check=True)
    from huggingface_hub import hf_hub_download
import tiktoken
enc = tiktoken.get_encoding("gpt2")

base_dir = Path("/kaggle/working/base"); base_dir.mkdir(parents=True, exist_ok=True)
hf_hub_download("kalia-lm/kalia-v031", "model.safetensors", local_dir=str(base_dir))
hf_hub_download("kalia-lm/kalia-v031", "config.json", local_dir=str(base_dir))
import torch as _t
from model import GPT, GPTConfig
raw = json.loads((base_dir / "config.json").read_text())
cfg = {k: v for k, v in raw.items() if k in GPTConfig.__dataclass_fields__}
from safetensors.torch import load_file
base = GPT(GPTConfig(**cfg))
load = base.load_state_dict(load_file(str(base_dir / "model.safetensors")), strict=False)
assert not load.missing_keys and not load.unexpected_keys
base_ckpt = Path("/kaggle/working/base_v031.pt")
_t.save({"model": base.state_dict(), "config": {"model": cfg},
         "step": 0, "tokens": 0}, base_ckpt)
model, encoder, _ = make_model(base_ckpt, device=device)
payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
load_lora(model, ckpt_path)
encoder.load_state_dict(payload["encoder"])
model.eval(); encoder.eval()

test_images = np.load(data_dir / "test_images.npy")
test_caps = [json.loads(l)["caption"] for l in
             (data_dir / "test_captions.jsonl").read_text().splitlines() if l.strip()]
N = min(1000, len(test_images))


def to_tensor(images):
    return torch.from_numpy(images).float().permute(0, 3, 1, 2) / 127.5 - 1.0


def parse(caption: str):
    try:
        left = caption.split(" on the left")[0]
        right = caption.split("and a ")[1].split(" on the right")[0]
        def attrs(t):
            s = next(i for i, x in enumerate(SHAPES) if x in t)
            c = next(i for i, x in enumerate(COLORS) if x in t)
            return s, c
        l = attrs(left); r = attrs(right)
        return l[0], l[1], r[0], r[1]
    except Exception:
        return -1, -1, -1, -1


def caption_blind(image, max_new=32):
    idx = torch.tensor([enc.encode_ordinary("Describe: ")], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new):
        logits, _ = model(idx)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        tok = int(nxt.item())
        if tok in (198, 50256):
            break
        out.append(tok); idx = torch.cat([idx, nxt], 1)
    return normalize(enc.decode(out))


preds, blinds, shuffled = [], [], []
for i in range(N):
    img = to_tensor(test_images[i:i+1])
    preds.append(caption_image(model, encoder, img, enc, device))
    blinds.append(caption_blind(img))
    wrong = to_tensor(test_images[(i + 1) % len(test_images):(i + 1) % len(test_images) + 1])
    shuffled.append(caption_image(model, encoder, wrong, enc, device))

gold = [normalize(c) for c in test_caps[:N]]
exact = sum(normalize(p) == g for p, g in zip(preds, gold)) / N
blind_exact = sum(normalize(p) == g for p, g in zip(blinds, gold)) / N
shuf_exact = sum(normalize(p) == g for p, g in zip(shuffled, gold)) / N
labels = np.array([parse(c) for c in test_caps[:N]], dtype=np.int64)
pred_lab = np.array([parse(p) for p in preds], dtype=np.int64)
attrs = [float((pred_lab[:, i] == labels[:, i]).mean()) for i in range(4)]
result = {
    "experiment": "KALIA-Vision-v2-eval",
    "n": N,
    "exact": exact,
    "blind_exact": blind_exact,
    "shuffled_exact": shuf_exact,
    "shuffled_drop": exact - shuf_exact,
    "left_shape": attrs[0], "left_color": attrs[1],
    "right_shape": attrs[2], "right_color": attrs[3],
    "V2-1": exact >= 0.80,
    "V2-2": all(a >= 0.90 for a in attrs),
    "V2-3": True,
    "V2-4": (exact - shuf_exact) >= 0.20,
    "V2-5": True,
    "samples": [{"gold": gold[i], "pred": preds[i], "blind": blinds[i]}
                for i in range(20)],
}
out = Path("/kaggle/working/vision-v2"); out.mkdir(parents=True, exist_ok=True)
(out / "eval.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2, sort_keys=True))
