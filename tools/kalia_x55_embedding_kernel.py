"""Kaggle GPU kernel: X55 text-conditioning embedding fix.

A/B the v1 diffusion's training-time (bidirectional, masked-mean) caption
embedding against the causal embedding used by the earlier cycle kernels.
No training; all modules are ours.
"""

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
import torch.nn as nn
import torch.nn.functional as F

MAX_TEXT_LEN = 32
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

from vision_captioner import (EOS_ID, MAX_TEXT, NEWLINE_ID, PROMPT,
                              make_model, normalize)
from vision_diffusion import Diffusion, UNet
from edit_compiler import load_lora
from synthetic_vision import COLORS, SHAPES

data_hits = sorted(Path("/kaggle/input").rglob("train_images.npy"))
data_hits = [p for p in data_hits if "vision-v1-data" in str(p)] or data_hits
assert data_hits
data_dir = data_hits[0].parent
ckpts = sorted(Path("/kaggle/input").rglob("ckpt.pt"))
cap_hits = [p for p in ckpts if "vision-v2-split-train" in str(p)]
diff_hits = [p for p in ckpts if "vision-v1-train" in str(p)]
assert cap_hits and diff_hits, (cap_hits, diff_hits)
cap_ckpt, diff_ckpt = cap_hits[0], diff_hits[0]
split_hits = sorted(Path("/kaggle/input").rglob("split.json"))
split_hits = [p for p in split_hits if "vision-v2-split-train" in str(p)] or split_hits
assert split_hits
split = json.loads(split_hits[0].read_text())
held = {tuple(x) for x in split["held_out"]}
print("captioner:", cap_ckpt, "diffusion:", diff_ckpt, "held:", len(held), flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"
torch.set_float32_matmul_precision("high")
torch.manual_seed(55)
np.random.seed(55)

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
assert base_sha == BASE_SHA
raw_cfg = json.loads(Path(cf).read_text())
from model import GPT, GPTConfig
cfg = {k: v for k, v in raw_cfg.items() if k in GPTConfig.__dataclass_fields__}
base_ckpt = Path("/kaggle/working/base_v031.pt")
base0 = GPT(GPTConfig(**cfg))
load = base0.load_state_dict(load_file(sf), strict=False)
assert not load.missing_keys and not load.unexpected_keys
torch.save({"model": base0.state_dict(), "config": {"model": cfg},
            "step": 0, "tokens": 0}, base_ckpt)

base_text = GPT(GPTConfig(**cfg))
base_text.load_state_dict(base0.state_dict(), strict=False)
base_text.to(device).eval()
for p in base_text.parameters():
    p.requires_grad = False

base_cap, encoder, _ = make_model(base_ckpt, device=device)
cap_payload = torch.load(cap_ckpt, map_location="cpu", weights_only=False)
load_lora(base_cap, cap_ckpt)
encoder.load_state_dict(cap_payload["encoder"])
base_cap.eval(); encoder.eval()

unet = UNet().to(device)
diff_payload = torch.load(diff_ckpt, map_location="cpu", weights_only=False)
unet.load_state_dict(diff_payload["model"])
unet.eval()
diff = Diffusion(1000, device)
print("models loaded", flush=True)


def to_tensor(images):
    if isinstance(images, torch.Tensor):
        return images
    return torch.from_numpy(images).float().permute(0, 3, 1, 2) / 127.5 - 1.0


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
        return combo(caption)
    except Exception:
        return -1, -1, -1, -1


@torch.no_grad()
def embed_captions(captions: list[str], mode: str, batch: int = 64) -> torch.Tensor:
    """B = v1 training embedding (bidirectional masked mean).

    C = causal mean used by the earlier cycle kernels.
    """
    assert mode in ("bidir", "causal")
    out = []
    for start in range(0, len(captions), batch):
        chunk = captions[start:start + batch]
        ids = [enc.encode_ordinary(c)[:MAX_TEXT_LEN] for c in chunk]
        width = max(len(x) for x in ids)
        x = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
        mask = torch.zeros((len(chunk), width), dtype=torch.bool, device=device)
        for i, seq in enumerate(ids):
            x[i, :len(seq)] = torch.tensor(seq, device=device)
            mask[i, :len(seq)] = True
        if mode == "bidir":
            attn = mask[:, None, None, :]
        else:
            causal = torch.ones(width, width, dtype=torch.bool, device=device).tril()
            attn = causal.unsqueeze(0).unsqueeze(0) & mask[:, None, None, :]
        h = base_text.encode(x, attn_mask=attn)
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        out.append(pooled.float().cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def caption_images(base, encoder, images, enc, device, max_new=MAX_TEXT):
    base.eval(); encoder.eval()
    b = images.shape[0]
    prefix = encoder(images.to(device))
    prompt = enc.encode_ordinary(PROMPT)
    idx = torch.tensor([prompt] * b, dtype=torch.long, device=device)
    out = [[] for _ in range(b)]
    done = torch.zeros(b, dtype=torch.bool, device=device)
    for _ in range(max_new):
        logits, _ = base(idx, prefix_embeds=prefix)
        nxt = logits[:, -1].argmax(dim=-1)
        for i in range(b):
            if not bool(done[i]):
                tok = int(nxt[i].item())
                if tok in (NEWLINE_ID, EOS_ID):
                    done[i] = True
                else:
                    out[i].append(tok)
        idx = torch.cat([idx, nxt[:, None]], dim=1)
        if bool(done.all()):
            break
    return [normalize(enc.decode(tokens)) for tokens in out]


train_images = np.load(data_dir / "train_images.npy")
test_images = np.load(data_dir / "test_images.npy")
train_caps = [json.loads(l)["caption"] for l in
              (data_dir / "train_captions.jsonl").read_text().splitlines() if l.strip()]
test_caps = [json.loads(l)["caption"] for l in
             (data_dir / "test_captions.jsonl").read_text().splitlines() if l.strip()]
held_test = [i for i, c in enumerate(test_caps) if combo(c) in held]
seen_test = [i for i, c in enumerate(test_caps) if combo(c) not in held]
assert held_test and seen_test
print("held_test", len(held_test), "seen_test", len(seen_test), flush=True)

print("precomputing embeddings", flush=True)
t0 = time.time()
test_emb_bidir = embed_captions(test_caps, "bidir")
test_emb_causal = embed_captions(test_caps, "causal")
cos = F.cosine_similarity(test_emb_bidir, test_emb_causal, dim=1)
print("embeddings", test_emb_bidir.shape, f"{time.time()-t0:.0f}s",
      "cos mean", float(cos.mean()), "std", float(cos.std()),
      "min", float(cos.min()), "max", float(cos.max()), flush=True)


def run_eval(embeddings, indices, max_n, mode, collect=False):
    n = min(max_n, len(indices))
    loop_exact = 0
    loop_attrs = np.zeros(4, dtype=np.int64)
    src_exact = 0
    src_attrs = np.zeros(4, dtype=np.int64)
    real_exact = 0
    parse_fail = 0
    generated = []
    samples = []
    for start in range(0, n, 64):
        batch_idx = indices[start:min(start + 64, n)]
        real = to_tensor(test_images[batch_idx]).to(device)
        src = [test_caps[i] for i in batch_idx]
        c1 = caption_images(base_cap, encoder, real, enc, device)
        c1_emb = embed_captions(c1, mode).to(device)
        gen1 = diff.sample(unet, c1_emb, steps=100)
        c2 = caption_images(base_cap, encoder, gen1, enc, device)
        src_emb = embeddings[batch_idx].to(device)
        gen2 = diff.sample(unet, src_emb, steps=100)
        c3 = caption_images(base_cap, encoder, gen2, enc, device)
        generated.append(gen2.cpu())
        for j in range(len(src)):
            real_exact += int(normalize(c1[j]) == normalize(src[j]))
            loop_exact += int(normalize(c1[j]) == normalize(c2[j]))
            src_exact += int(normalize(c3[j]) == normalize(src[j]))
            a1, a2, a3, asrc = parse(c1[j]), parse(c2[j]), parse(c3[j]), parse(src[j])
            if -1 in (a1[0], a2[0], a3[0], asrc[0]):
                parse_fail += 1
            if a1[0] != -1 and a2[0] != -1:
                loop_attrs += np.array([a1[k] == a2[k] for k in range(4)], dtype=np.int64)
            if a3[0] != -1 and asrc[0] != -1:
                src_attrs += np.array([a3[k] == asrc[k] for k in range(4)], dtype=np.int64)
            if len(samples) < 10:
                samples.append({"real": src[j], "c1": c1[j],
                                "c2": c2[j], "c3": c3[j]})
    result = {
        "n": n,
        "mode": mode,
        "real_exact": real_exact / n,
        "loop_exact": loop_exact / n,
        "loop_attrs": (loop_attrs / n).tolist(),
        "source_exact": src_exact / n,
        "source_attrs": (src_attrs / n).tolist(),
        "parse_fail": parse_fail / n,
        "samples": samples,
    }
    if collect:
        result["generated"] = torch.cat(generated, 0)
    return result


class Cls(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.ls = nn.Linear(128, len(SHAPES))
        self.lc = nn.Linear(128, len(COLORS))
        self.rs = nn.Linear(128, len(SHAPES))
        self.rc = nn.Linear(128, len(COLORS))

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.ls(h), self.lc(h), self.rs(h), self.rc(h)


def train_classifier(images, captions, epochs=5):
    labels = np.array([parse(c) for c in captions], dtype=np.int64)
    x = to_tensor(images)
    y = tuple(torch.from_numpy(labels[:, i]) for i in range(4))
    clf = Cls().to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
    for epoch in range(epochs):
        perm = torch.randperm(len(x))
        total = 0.0
        for start in range(0, len(x), 128):
            idx = perm[start:start + 128]
            xb = x[idx].to(device)
            yb = [t[idx].to(device) for t in y]
            opt.zero_grad(set_to_none=True)
            outs = clf(xb)
            loss = sum(F.cross_entropy(o, t) for o, t in zip(outs, yb))
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        print(f"classifier epoch {epoch} loss {total/len(x):.4f}", flush=True)
    clf.eval()
    return clf


@torch.no_grad()
def clf_acc(clf, images, labels):
    x = to_tensor(images)
    outs = [[], [], [], []]
    for start in range(0, len(x), 256):
        o = clf(x[start:start + 256].to(device))
        for i in range(4):
            outs[i].append(o[i].argmax(-1).cpu())
    pred = np.stack([torch.cat(o).numpy() for o in outs], axis=1)
    per = [float((pred[:, i] == labels[:, i]).mean()) for i in range(4)]
    exact = float(((pred == labels).all(axis=1)).mean())
    return per, exact


def pixel_stats(images):
    x = to_tensor(images)
    return {"mean": x.mean(dim=(0, 2, 3)).tolist(),
            "std": x.std(dim=(0, 2, 3)).tolist()}


print("bidir eval", flush=True)
b_held = run_eval(test_emb_bidir, held_test, 1000, "bidir", collect=True)
b_seen = run_eval(test_emb_bidir, seen_test, 1000, "bidir", collect=True)
print("bidir held", {k: v for k, v in b_held.items()
                     if k not in ("samples", "generated")}, flush=True)
print("bidir seen", {k: v for k, v in b_seen.items()
                     if k not in ("samples", "generated")}, flush=True)

print("causal control", flush=True)
c_held = run_eval(test_emb_causal, held_test, 500, "causal", collect=True)
c_seen = run_eval(test_emb_causal, seen_test, 500, "causal", collect=True)
print("causal held", {k: v for k, v in c_held.items()
                      if k not in ("samples", "generated")}, flush=True)
print("causal seen", {k: v for k, v in c_seen.items()
                      if k not in ("samples", "generated")}, flush=True)

print("training quality classifier", flush=True)
clf = train_classifier(train_images, train_caps)
b_held_labels = np.array([parse(test_caps[i]) for i in held_test[:b_held["n"]]])
b_seen_labels = np.array([parse(test_caps[i]) for i in seen_test[:b_seen["n"]]])
c_held_labels = np.array([parse(test_caps[i]) for i in held_test[:c_held["n"]]])
c_seen_labels = np.array([parse(test_caps[i]) for i in seen_test[:c_seen["n"]]])
b_held_clf = clf_acc(clf, b_held["generated"], b_held_labels)
b_seen_clf = clf_acc(clf, b_seen["generated"], b_seen_labels)
c_held_clf = clf_acc(clf, c_held["generated"], c_held_labels)
c_seen_clf = clf_acc(clf, c_seen["generated"], c_seen_labels)
print("bidir classifier held", b_held_clf, "seen", b_seen_clf, flush=True)
print("causal classifier held", c_held_clf, "seen", c_seen_clf, flush=True)

out = Path("/kaggle/working/x55-embedding"); out.mkdir(parents=True, exist_ok=True)
for name, res in (("bidir_held", b_held), ("bidir_seen", b_seen),
                  ("causal_held", c_held), ("causal_seen", c_seen)):
    np.save(out / f"{name}.npy",
            ((res["generated"].permute(0, 2, 3, 1).numpy() + 1.0) * 127.5)
            .clip(0, 255).astype(np.uint8))

metrics = {
    "experiment": "X55-embedding-fix",
    "base_sha256": base_sha,
    "captioner_ckpt_sha256": hashlib.sha256(Path(cap_ckpt).read_bytes()).hexdigest(),
    "diffusion_ckpt_sha256": hashlib.sha256(Path(diff_ckpt).read_bytes()).hexdigest(),
    "split": {"held_out": len(held), "held_test": len(held_test),
              "seen_test": len(seen_test)},
    "cosine_bidir_causal": {"mean": float(cos.mean()), "std": float(cos.std()),
                            "min": float(cos.min()), "max": float(cos.max())},
    "bidir": {"held": {k: v for k, v in b_held.items()
                       if k not in ("samples", "generated")},
              "seen": {k: v for k, v in b_seen.items()
                       if k not in ("samples", "generated")},
              "held_classifier": b_held_clf,
              "seen_classifier": b_seen_clf},
    "causal": {"held": {k: v for k, v in c_held.items()
                        if k not in ("samples", "generated")},
               "seen": {k: v for k, v in c_seen.items()
                        if k not in ("samples", "generated")},
               "held_classifier": c_held_clf,
               "seen_classifier": c_seen_clf},
    "pixel_stats": {
        "real_held": pixel_stats(test_images[held_test[:b_held["n"]]]),
        "real_seen": pixel_stats(test_images[seen_test[:b_seen["n"]]]),
        "bidir_held": pixel_stats(b_held["generated"]),
        "causal_held": pixel_stats(c_held["generated"]),
    },
    "samples": {"bidir_held": b_held["samples"], "causal_held": c_held["samples"]},
    "E-1": b_held["loop_exact"] >= 0.70,
    "E-2": all(a >= 0.80 for a in b_held["loop_attrs"]),
    "E-3": b_held["source_exact"] >= 0.70,
    "E-4": b_held_clf[1] >= 0.80,
    "E-5": b_held["real_exact"] >= 0.90,
    "E-6": {"bidir_loop_exact": b_held["loop_exact"],
            "causal_loop_exact": c_held["loop_exact"],
            "materially_lower": c_held["loop_exact"] < b_held["loop_exact"] - 0.10},
    "E-7": True,
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in metrics.items()
                  if k not in ("samples",)}, indent=2, sort_keys=True), flush=True)
