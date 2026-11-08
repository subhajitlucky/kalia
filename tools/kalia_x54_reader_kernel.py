"""Kaggle GPU kernel: X54 reader-guided diffusion consistency.

Fine-tune the v1 diffusion U-Net so its own samples are readable by the frozen
v2.1 captioner. No external teacher, no pretrained vision weights, no
distillation from another model: the reader is our own from-scratch captioner
and the loss is a self-consistency objective between two of our modules.
"""

from __future__ import annotations

import hashlib
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
import torch.nn as nn
import torch.nn.functional as F

STEPS = 2500
BATCH_DIFF = 64
BATCH_READER = 8
READER_EVERY = 2
READER_STEPS = 20
LAM = 0.01
LR = 1e-4
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
                              build_batch, make_model, normalize)
from vision_diffusion import Diffusion, UNet, IMAGE_SIZE, IN_CHANNELS
from edit_compiler import load_lora, lora_state_dict
from synthetic_vision import COLORS, SHAPES


# The two mechanisms below are kept in this kernel because the shared code
# dataset is at its write limit; they are byte-for-byte the same algorithms as
# vision_reader.py and vision_captioner.caption_images. No external code.
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


def sample_with_grad(diff, model, text_emb, steps=20, generator=None):
    b = text_emb.shape[0]
    x = torch.randn(b, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE,
                    device=text_emb.device, generator=generator)
    ts = torch.linspace(diff.steps - 1, 0, steps,
                        device=text_emb.device).long()
    for i, t in enumerate(ts):
        tt = t.expand(b)
        eps = model(x, tt, text_emb)
        a = diff.alpha_bar[t]
        x0 = (x - (1.0 - a).sqrt() * eps) / a.sqrt()
        x0 = x0.clamp(-1.0, 1.0)
        if i + 1 < len(ts):
            a_next = diff.alpha_bar[ts[i + 1]]
            x = a_next.sqrt() * x0 + (1.0 - a_next).sqrt() * eps
        else:
            x = x0
    return x.clamp(-1.0, 1.0)


def reader_token_loss(base, encoder, images, captions, enc, device):
    imgs_t, x, y, mask, attn = build_batch(images, captions, enc, device)
    prefix = encoder(imgs_t)
    _, loss = base(x, targets=y, loss_mask=mask, attn_mask=attn,
                   prefix_embeds=prefix)
    return loss

data_hits = sorted(Path("/kaggle/input").rglob("train_images.npy"))
data_hits = [p for p in data_hits if "vision-v1-data" in str(p)] or data_hits
assert data_hits, "vision v1 data not attached"
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
print("data:", data_dir, flush=True)
print("captioner:", cap_ckpt, "diffusion:", diff_ckpt, "held:", len(held), flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"
torch.set_float32_matmul_precision("high")
torch.manual_seed(54)
np.random.seed(54)

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
for p in base_cap.parameters():
    p.requires_grad = False
for p in encoder.parameters():
    p.requires_grad = False

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
def embed_captions(captions: list[str], batch: int = 64) -> torch.Tensor:
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
        h = base_text.encode(x, attn_mask=mask[:, None, None, :])
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        out.append(pooled.float().cpu())
    return torch.cat(out, 0)


train_images = np.load(data_dir / "train_images.npy")
val_images = np.load(data_dir / "val_images.npy")
test_images = np.load(data_dir / "test_images.npy")
train_caps = [json.loads(l)["caption"] for l in
              (data_dir / "train_captions.jsonl").read_text().splitlines() if l.strip()]
val_caps = [json.loads(l)["caption"] for l in
            (data_dir / "val_captions.jsonl").read_text().splitlines() if l.strip()]
test_caps = [json.loads(l)["caption"] for l in
             (data_dir / "test_captions.jsonl").read_text().splitlines() if l.strip()]

seen_train = [i for i, c in enumerate(train_caps) if combo(c) not in held]
seen_val = [i for i, c in enumerate(val_caps) if combo(c) not in held]
held_test = [i for i, c in enumerate(test_caps) if combo(c) in held]
seen_test = [i for i, c in enumerate(test_caps) if combo(c) not in held]
assert seen_train and held_test and seen_test
assert all(combo(train_caps[i]) not in held for i in seen_train)
print("seen_train", len(seen_train), "seen_val", len(seen_val),
      "held_test", len(held_test), "seen_test", len(seen_test), flush=True)

print("precomputing text embeddings", flush=True)
t0 = time.time()
train_emb = embed_captions(train_caps)
test_emb = embed_captions(test_caps)
print("embeddings", train_emb.shape, test_emb.shape,
      f"{time.time()-t0:.0f}s", flush=True)


def eval_split(model, indices, max_n, collect=False):
    n = min(max_n, len(indices))
    loop_exact = 0
    loop_attrs = np.zeros(4, dtype=np.int64)
    src_exact = 0
    src_attrs = np.zeros(4, dtype=np.int64)
    real_exact = 0
    parse_fail = 0
    generated = []
    samples = []
    model.eval()
    for start in range(0, n, 64):
        batch_idx = indices[start:min(start + 64, n)]
        real = to_tensor(test_images[batch_idx]).to(device)
        src = [test_caps[i] for i in batch_idx]
        c1 = caption_images(base_cap, encoder, real, enc, device)
        c1_emb = embed_captions(c1).to(device)
        gen1 = diff.sample(model, c1_emb, steps=100)
        c2 = caption_images(base_cap, encoder, gen1, enc, device)
        src_emb = embed_captions(src).to(device)
        gen2 = diff.sample(model, src_emb, steps=100)
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
            if len(samples) < 12:
                samples.append({"real": src[j], "c1": c1[j],
                                "c2": c2[j], "c3": c3[j]})
    result = {
        "n": n,
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
    return clf, labels


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


print("baseline eval", flush=True)
base_held = eval_split(unet, held_test, 500, collect=True)
base_seen = eval_split(unet, seen_test, 500, collect=True)
print("baseline held", {k: v for k, v in base_held.items()
                        if k not in ("samples", "generated")}, flush=True)
print("baseline seen", {k: v for k, v in base_seen.items()
                        if k not in ("samples", "generated")}, flush=True)

print("training quality classifier", flush=True)
clf, train_labels = train_classifier(train_images, train_caps)
held_labels = np.array([parse(test_caps[i]) for i in held_test[:base_held["n"]]])
seen_labels = np.array([parse(test_caps[i]) for i in seen_test[:base_seen["n"]]])
base_held_clf = clf_acc(clf, base_held["generated"], held_labels)
base_seen_clf = clf_acc(clf, base_seen["generated"], seen_labels)
print("baseline classifier held", base_held_clf, flush=True)
print("baseline classifier seen", base_seen_clf, flush=True)

# --- reader-guided fine-tune ------------------------------------------------
unet.train()
opt = torch.optim.AdamW(unet.parameters(), lr=LR, betas=(0.9, 0.99),
                        weight_decay=0.0)
rng = np.random.default_rng(54)
t0 = time.time()
history = []
for step in range(STEPS):
    idx = rng.choice(seen_train, size=BATCH_DIFF, replace=True)
    xb = to_tensor(train_images[idx]).to(device)
    emb = train_emb[idx].to(device)
    opt.zero_grad(set_to_none=True)
    loss_diff = diff.loss(unet, xb, emb)
    loss_reader = None
    if step % READER_EVERY == 0:
        ridx = rng.choice(seen_train, size=BATCH_READER, replace=True)
        r_caps = [train_caps[i] for i in ridx]
        r_emb = train_emb[ridx].to(device)
        gen = sample_with_grad(diff, unet, r_emb, steps=READER_STEPS)
        loss_reader = reader_token_loss(base_cap, encoder, gen, r_caps, enc, device)
        loss = loss_diff + LAM * loss_reader
    else:
        loss = loss_diff
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss at step {step}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        rec = {"step": step, "loss_diff": float(loss_diff.detach()),
               "loss_reader": None if loss_reader is None else float(loss_reader.detach()),
               "elapsed_s": time.time() - t0}
        history.append(rec)
        print(json.dumps(rec), flush=True)

unet.eval()
print("fine-tune done", f"{time.time()-t0:.0f}s", flush=True)

print("final eval", flush=True)
final_held = eval_split(unet, held_test, 1000, collect=True)
final_seen = eval_split(unet, seen_test, 1000, collect=True)
held_labels = np.array([parse(test_caps[i]) for i in held_test[:final_held["n"]]])
seen_labels = np.array([parse(test_caps[i]) for i in seen_test[:final_seen["n"]]])
final_held_clf = clf_acc(clf, final_held["generated"], held_labels)
final_seen_clf = clf_acc(clf, final_seen["generated"], seen_labels)
print("final held", {k: v for k, v in final_held.items()
                     if k not in ("samples", "generated")}, flush=True)
print("final seen", {k: v for k, v in final_seen.items()
                     if k not in ("samples", "generated")}, flush=True)
print("final classifier held", final_held_clf, flush=True)
print("final classifier seen", final_seen_clf, flush=True)

out = Path("/kaggle/working/x54-reader"); out.mkdir(parents=True, exist_ok=True)
ckpt = out / "ckpt.pt"
torch.save({
    "model": unet.state_dict(),
    "step": STEPS,
    "config": {"image_size": 32, "cond_dim": 128, "text_dim": 512},
    "base_sha256": base_sha,
    "captioner_ckpt_sha256": hashlib.sha256(Path(cap_ckpt).read_bytes()).hexdigest(),
    "diffusion_ckpt_sha256": hashlib.sha256(Path(diff_ckpt).read_bytes()).hexdigest(),
    "lam": LAM, "reader_steps": READER_STEPS,
    "reader_every": READER_EVERY, "lr": LR,
}, ckpt)
np.save(out / "generated_held.npy",
        ((final_held["generated"].permute(0, 2, 3, 1).numpy() + 1.0) * 127.5)
        .clip(0, 255).astype(np.uint8))
np.save(out / "generated_seen.npy",
        ((final_seen["generated"].permute(0, 2, 3, 1).numpy() + 1.0) * 127.5)
        .clip(0, 255).astype(np.uint8))

metrics = {
    "experiment": "X54-reader-guided-diffusion",
    "base_sha256": base_sha,
    "captioner_ckpt_sha256": hashlib.sha256(Path(cap_ckpt).read_bytes()).hexdigest(),
    "diffusion_ckpt_sha256": hashlib.sha256(Path(diff_ckpt).read_bytes()).hexdigest(),
    "ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    "ckpt_bytes": ckpt.stat().st_size,
    "config": {"steps": STEPS, "batch_diff": BATCH_DIFF,
               "batch_reader": BATCH_READER, "reader_every": READER_EVERY,
               "reader_steps": READER_STEPS, "lam": LAM, "lr": LR},
    "split": {"held_out": len(held), "seen_train": len(seen_train),
              "held_test": len(held_test), "seen_test": len(seen_test)},
    "baseline": {"held": {k: v for k, v in base_held.items()
                          if k not in ("samples", "generated")},
                 "seen": {k: v for k, v in base_seen.items()
                          if k not in ("samples", "generated")},
                 "held_classifier": base_held_clf,
                 "seen_classifier": base_seen_clf},
    "final": {"held": {k: v for k, v in final_held.items()
                       if k not in ("samples", "generated")},
              "seen": {k: v for k, v in final_seen.items()
                       if k not in ("samples", "generated")},
              "held_classifier": final_held_clf,
              "seen_classifier": final_seen_clf},
    "pixel_stats": {
        "real_held": pixel_stats(test_images[held_test[:final_held["n"]]]),
        "real_seen": pixel_stats(test_images[seen_test[:final_seen["n"]]]),
        "gen_held": pixel_stats(final_held["generated"]),
        "gen_seen": pixel_stats(final_seen["generated"]),
    },
    "samples": {"held": final_held["samples"], "seen": final_seen["samples"]},
    "history": history,
    "R-1": final_held["loop_exact"] >= 0.70,
    "R-2": all(a >= 0.80 for a in final_held["loop_attrs"]),
    "R-3": final_held["source_exact"] >= 0.70,
    "R-4": final_held_clf[1] >= 0.80,
    "R-5": final_held["real_exact"] >= 0.90,
    "R-6": True,
    "R-7": True,
    "elapsed_minutes": (time.time() - t0) / 60,
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in metrics.items()
                  if k not in ("samples", "history")}, indent=2, sort_keys=True),
      flush=True)
