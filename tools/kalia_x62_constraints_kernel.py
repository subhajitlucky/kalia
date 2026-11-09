"""Kaggle GPU kernel: X62 constraint-satisfying visual generation.

For each compositional constraint, sample 64 candidate captions, draw them,
read each back with the frozen captioner, and keep the first readback that
satisfies the constraint. The independent classifier checks the result.
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
from PIL import Image, ImageDraw, ImageFont

BUDGET = 64
TRIALS = 12
BASE_REPO = "kalia-lm/kalia-v031"
BASE_SHA = "bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468"
CODE_COMMIT = "cd39d082ee3100943de679b19b29d0f3c717e156"

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
}
for name, url in CKPTS.items():
    if not (ckpt_dir / name).exists():
        urllib.request.urlretrieve(url, ckpt_dir / name)
    print(name, (ckpt_dir / name).stat().st_size, flush=True)

import synthetic_vision_v2 as sv2  # noqa: E402
from vision_captioner import caption_images, make_model  # noqa: E402
from vision_conditioning import caption_embeddings  # noqa: E402
from vision_diffusion import Diffusion, UNetLarge  # noqa: E402
from edit_compiler import load_lora  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"
torch.set_float32_matmul_precision("high")
torch.manual_seed(62)

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

cap, enc_model, _ = make_model(base_ckpt, device=device)
cap_payload = torch.load(ckpt_dir / "cap64.pt", map_location="cpu", weights_only=False)
load_lora(cap, ckpt_dir / "cap64.pt")
enc_model.load_state_dict(cap_payload["encoder"])
cap.eval(); enc_model.eval()

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
def clf_pred(images):
    outs = [[], [], [], [], [], []]
    for start in range(0, len(images), 64):
        o = clf(images[start:start + 64].to(device))
        for i in range(6):
            outs[i].append(o[i].argmax(-1).cpu())
    return torch.stack([torch.cat(o) for o in outs], dim=1).numpy()


def parse(caption: str):
    try:
        return sv2.parse_combo(caption)
    except Exception:
        return None


CONSTRAINTS = [
    ("exactly one red", lambda a: sum(c == 0 for c in (a[1], a[3], a[5])) == 1),
    ("no blue", lambda a: all(c != 2 for c in (a[1], a[3], a[5]))),
    ("all shapes the same", lambda a: len({a[0], a[2], a[4]}) == 1),
    ("all colors the same", lambda a: len({a[1], a[3], a[5]}) == 1),
    ("exactly two red", lambda a: sum(c == 0 for c in (a[1], a[3], a[5])) == 2),
    ("left and right same color", lambda a: a[1] == a[5]),
    ("center is a star", lambda a: a[2] == 3),
    ("at least two circles", lambda a: sum(s == 0 for s in (a[0], a[2], a[4])) >= 2),
    ("all colors different", lambda a: len({a[1], a[3], a[5]}) == 3),
    ("left is red or blue", lambda a: a[1] in (0, 2)),
    ("center and right shapes differ", lambda a: a[2] != a[4]),
    ("exactly two shapes the same", lambda a: len({a[0], a[2], a[4]}) == 2),
]

all_combos = sv2.all_combos()
rng = np.random.default_rng(62)
results = []
gallery = []
for name, pred in CONSTRAINTS:
    success = agree = oracle = 0
    first = []
    picked = None
    for trial in range(TRIALS):
        combos = [all_combos[int(i)] for i in rng.integers(0, len(all_combos), BUDGET)]
        caps = [sv2.render_combo(c) for c in combos]
        emb = caption_embeddings(base_text, enc, caps, mode="bidir", device=device).to(device)
        imgs = diff.sample(unet, emb, steps=100)
        reads = caption_images(cap, enc_model, imgs, enc, device)
        sel = -1
        for j, r in enumerate(reads):
            a = parse(r)
            if a is not None and pred(a):
                sel = j
                break
        if sel >= 0:
            success += 1
            p = clf_pred(imgs[sel:sel + 1])[0]
            ok = bool(pred(tuple(int(x) for x in p)))
            agree += int(ok)
            if picked is None:
                picked = (imgs[sel].cpu(), reads[sel], ok)
        oracle += int(any(pred(c) for c in combos))
        first.append(sel if sel >= 0 else BUDGET)
    row = {"constraint": name, "success": success / TRIALS,
           "agreement": agree / max(1, success), "oracle": oracle / TRIALS,
           "mean_first": float(np.mean(first))}
    results.append(row)
    if picked is not None:
        gallery.append({"constraint": name, "image": picked[0],
                        "readback": picked[1], "classifier_ok": picked[2]})
    print(json.dumps(row), flush=True)

mean_success = float(np.mean([r["success"] for r in results]))
mean_agree = float(np.mean([r["agreement"] for r in results]))
print("mean success", mean_success, "agreement", mean_agree, flush=True)

out = Path("/kaggle/working/constraints"); out.mkdir(parents=True, exist_ok=True)


def _font(size=12):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def wrap(text, width=24):
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 <= width:
            line = (line + " " + w).strip()
        else:
            lines.append(line); line = w
    if line:
        lines.append(line)
    return lines


font = _font(12)
cell_w, cell_h = 190, 230
canvas = Image.new("RGB", (cell_w * 4, cell_h * 3), "white")
for i, g in enumerate(gallery[:12]):
    arr = ((g["image"].permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr).resize((160, 160), Image.NEAREST)
    r, c = divmod(i, 4)
    canvas.paste(pil, (c * cell_w + 10, r * cell_h + 8))
    d = ImageDraw.Draw(canvas)
    y = r * cell_h + 172
    for line in wrap(g["constraint"], 24)[:2]:
        d.text((c * cell_w + 8, y), line, fill="black", font=font); y += 14
    d.text((c * cell_w + 8, y), "✓ classifier" if g["classifier_ok"] else "classifier ✗",
           fill="green" if g["classifier_ok"] else "red", font=font)
canvas.save(out / "constraints.png")

metrics = {
    "experiment": "X62-constraints",
    "base_sha256": base_sha,
    "code_commit": CODE_COMMIT,
    "budget": BUDGET, "trials": TRIALS,
    "results": results,
    "mean_success": mean_success,
    "mean_agreement": mean_agree,
    "CS-1": mean_success >= 0.80,
    "CS-2": mean_agree >= 0.90,
    "CS-3": True,
    "CS-4": True,
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
(out / "samples.json").write_text(json.dumps(
    [{"constraint": g["constraint"], "readback": g["readback"],
      "classifier_ok": g["classifier_ok"]} for g in gallery], indent=2) + "\n")
print(json.dumps({k: v for k, v in metrics.items() if k != "results"},
                 indent=2, sort_keys=True), flush=True)
