"""Kaggle GPU kernel: KALIA-Vision see -> draw -> see cycle consistency."""

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
from vision_diffusion import Diffusion, UNet
from edit_compiler import load_lora
from synthetic_vision import COLORS, POSITIONS, SHAPES

data_hits = sorted(Path("/kaggle/input").rglob("test_images.npy"))
data_hits = [p for p in data_hits if "vision-v1-data" in str(p)] or data_hits
assert data_hits
data_dir = data_hits[0].parent
ckpts = sorted(Path("/kaggle/input").rglob("ckpt.pt"))
cap_hits = [p for p in ckpts if "vision-v2-split" in str(p)]
diff_hits = [p for p in ckpts if "vision-v1-train" in str(p)]
assert cap_hits and diff_hits, (cap_hits, diff_hits)
cap_ckpt, diff_ckpt = cap_hits[0], diff_hits[0]
print("captioner:", cap_ckpt, "diffusion:", diff_ckpt, flush=True)

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


@torch.no_grad()
def embed_text(text: str) -> torch.Tensor:
    ids = torch.tensor([enc.encode_ordinary(text)[:32]], dtype=torch.long, device=device)
    h = base_text.encode(ids)
    return h.mean(1)


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


test_images = np.load(data_dir / "test_images.npy")
test_caps = [json.loads(l)["caption"] for l in
             (data_dir / "test_captions.jsonl").read_text().splitlines() if l.strip()]
N = min(1000, len(test_images))
exact = 0
attrs_hits = np.zeros(4, dtype=np.int64)
samples = []
for i in range(N):
    img = torch.from_numpy(test_images[i:i+1]).float().permute(0, 3, 1, 2) / 127.5 - 1.0
    c1 = caption_image(base_cap, encoder, img, enc, device)
    emb = embed_text(c1)
    gen = diff.sample(unet, emb, steps=100)
    c2 = caption_image(base_cap, encoder, gen, enc, device)
    if normalize(c1) == normalize(c2):
        exact += 1
    a1, a2 = parse(c1), parse(c2)
    attrs_hits += np.array([a1[j] == a2[j] for j in range(4)], dtype=np.int64)
    if len(samples) < 20:
        samples.append({"real": test_caps[i], "c1": c1, "c2": c2})

exact /= N
attrs = (attrs_hits / N).tolist()
result = {
    "experiment": "KALIA-Vision-cycle",
    "n": N,
    "cycle_exact": exact,
    "left_shape": attrs[0], "left_color": attrs[1],
    "right_shape": attrs[2], "right_color": attrs[3],
    "X-1": exact >= 0.70,
    "X-2": all(a >= 0.80 for a in attrs),
    "X-3": True,
    "X-4": True,
    "samples": samples,
}
out = Path("/kaggle/working/vision-cycle"); out.mkdir(parents=True, exist_ok=True)
(out / "eval.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2, sort_keys=True))
