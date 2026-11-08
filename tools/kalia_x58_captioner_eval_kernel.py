"""Kaggle GPU kernel: X58 64x64 captioner evaluation (recovered checkpoint).

Loads the checkpoint saved before the failed eval, fixes the last-batch
truncation, and measures held-out / seen exact with blind and shuffled controls.
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

TEST_N = 1000
BASE_REPO = "kalia-lm/kalia-v031"
BASE_SHA = "bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468"
CODE_COMMIT = "703d682126c8246dd5f240efc00abf5b850a177d"

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
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/subhajitlucky/kalia/{CODE_COMMIT}/{name}",
        work / name,
    )
import synthetic_vision_v2 as sv  # noqa: E402
from vision_captioner import (caption_images, make_model, normalize)  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"
torch.set_float32_matmul_precision("high")

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

ckpt_path = Path("/kaggle/working/ckpt_captioner.pt")
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/subhajitlucky/kalia/{CODE_COMMIT}/docs/results/kalia-vision-64/ckpt_captioner.pt",
    ckpt_path,
)
print("ckpt", hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:16], flush=True)

model, encoder, _ = make_model(base_ckpt, device=device)
payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
encoder.load_state_dict(payload["encoder"])
model.load_state_dict(payload["lora"], strict=False)
model.eval(); encoder.eval()

print("generating test data", flush=True)
seen, held = sv.split_combos()
test_images, test_recs = sv.generate(5000, 23, held)
seen_images, seen_recs = sv.generate(5000, 24, seen)
test_caps = [r["caption"] for r in test_recs]
seen_caps = [r["caption"] for r in seen_recs]


def to_tensor(images):
    return torch.from_numpy(images).float().permute(0, 3, 1, 2) / 127.5 - 1.0


def parse(caption: str):
    try:
        return sv.parse_combo(caption)
    except Exception:
        return (-1,) * 6


def run_eval(images, caps, n):
    n = min(n, len(images))
    preds = []
    for start in range(0, n, 64):
        batch = to_tensor(images[start:min(start + 64, n)])
        preds += caption_images(model, encoder, batch, enc, device)
    exact = sum(normalize(p) == normalize(c) for p, c in zip(preds, caps[:n])) / n
    labels = np.array([parse(c) for c in caps[:n]], dtype=np.int64)
    pred_lab = np.array([parse(p) for p in preds], dtype=np.int64)
    attrs = [float((pred_lab[:, i] == labels[:, i]).mean()) for i in range(6)]
    return exact, attrs, preds


held_exact, held_attrs, held_preds = run_eval(test_images, test_caps, TEST_N)
seen_exact, seen_attrs, _ = run_eval(seen_images, seen_caps, TEST_N)
print("held", held_exact, held_attrs, "seen", seen_exact, seen_attrs, flush=True)

blind, shuf = [], []
for i in range(200):
    img = to_tensor(test_images[i:i + 1])
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

out = Path("/kaggle/working/vision64-captioner-eval"); out.mkdir(parents=True, exist_ok=True)
samples = [{"gold": test_caps[i], "pred": held_preds[i],
            "ok": normalize(held_preds[i]) == normalize(test_caps[i])}
           for i in range(20)]
from PIL import Image
grid = test_images[:64]
canvas = np.zeros((8 * 64, 8 * 64, 3), dtype=np.uint8)
for i in range(64):
    r, c = divmod(i, 8)
    canvas[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = grid[i]
Image.fromarray(canvas).resize((512, 512), Image.NEAREST).save(out / "grid.png")

meta = {
    "experiment": "X58-vision-64-captioner-eval",
    "base_sha256": base_sha,
    "code_commit": CODE_COMMIT,
    "ckpt_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
    "ckpt_bytes": ckpt_path.stat().st_size,
    "held_combos": len(held), "seen_combos": len(seen),
    "held_exact": held_exact, "seen_exact": seen_exact,
    "held_attrs": held_attrs, "seen_attrs": seen_attrs,
    "blind_exact": blind_exact, "shuffled_exact": shuf_exact,
    "V64-1": held_exact >= 0.90,
    "V64-2": seen_exact >= 0.95,
    "samples": samples,
}
(out / "metrics.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in meta.items() if k != "samples"},
                 indent=2, sort_keys=True), flush=True)
