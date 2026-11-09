"""Kaggle GPU kernel: X64 150M depth-expand smoke.

Builds the 147.8M 38-layer checkpoint by interpolating v0.3.1's 10 layers,
checks the expanded init val loss against the 2.7579 + 0.15 gate, then runs a
150-minute smoke of the registered continued-pretrain recipe.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
CODE_COMMIT = "9cf25a9004bfad86892d9cdc6732e27b1ba9598e"
MAX_MINUTES = 150
INIT_GATE = 2.7579 + 0.15

hits = sorted(glob.glob("/kaggle/input/**/train.py", recursive=True))
preferred = [p for p in hits if "/kalia-code-dev/" in p]
assert preferred, f"kalia-code-dev not attached: {hits}"
code_root = os.path.dirname(preferred[0])
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("train.py", "model.py", "optim.py", "data.py", "eval_val.py",
            "eval_reversibility.py"):
    assert os.path.exists(req), f"{req} missing from code snapshot"
print("code:", code_root, flush=True)

RAW = f"https://raw.githubusercontent.com/subhajitlucky/kalia/{CODE_COMMIT}"
for name in ("configs/kalia-v150.yaml", "configs/kalia-v150-smoke.yaml",
             "tools/depth_expand.py"):
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{RAW}/{name}", name)
print("fetched X64 config/expander", flush=True)

train_hits = sorted(glob.glob("/kaggle/input/**/train.bin", recursive=True))
preferred_train = [p for p in train_hits if "/kalia-prep-v2b/" in p]
train_hits = preferred_train or train_hits
assert train_hits, "attach subhajitlucky/kalia-prep-v2b for train.bin"
data_dir = os.path.dirname(train_hits[0])
train_bin = train_hits[0]
val_bin = os.path.join(data_dir, "val.bin")
assert os.path.exists(val_bin), f"val.bin missing next to {train_bin}"
print("data:", data_dir, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def val_loss_from_md(path):
    text = Path(path).read_text()
    m = re.search(r"Val loss:\s*\*\*([0-9.]+)\*\*", text)
    assert m, text
    return float(m.group(1))


# --- expand v0.3.1 10 -> 38 layers ------------------------------------------
init_ckpt = "/kaggle/working/kalia-v150-init.pt"
subprocess.run([sys.executable, "tools/depth_expand.py",
                "--base-repo", "kalia-lm/kalia-v031",
                "--out", init_ckpt, "--layers", "38"], check=True)

from model import GPT, GPTConfig  # noqa: E402
payload = torch.load(init_ckpt, map_location="cpu", weights_only=False)
params = sum(p.numel() for p in GPT(GPTConfig(**payload["config"]["model"])).parameters())
print("expanded params:", params, flush=True)

# --- init val gate -----------------------------------------------------------
init_md = "/kaggle/working/init_val.md"
subprocess.run([sys.executable, "eval_val.py", "--ckpt", init_ckpt,
                "--val-bin", val_bin, "--batches", "20", "--batch-size", "8",
                "--device", "cuda", "--out", init_md], check=True)
init_val = val_loss_from_md(init_md)
print("init val", init_val, "gate", INIT_GATE, flush=True)
assert init_val <= INIT_GATE, (
    f"expanded init val {init_val:.4f} exceeds gate {INIT_GATE:.4f}; "
    "the depth interpolation is not preserving the function"
)

# --- smoke train -------------------------------------------------------------
cmd = ["train.py", "--config", "configs/kalia-v150-smoke.yaml",
       "--data-dir", data_dir, "--out-dir", "/kaggle/working/out",
       "--resume-from", init_ckpt, "--seed", "1502",
       "--max-minutes", str(MAX_MINUTES)]
if torch.cuda.device_count() >= 2:
    cmd = [sys.executable, "-m", "torch.distributed.run",
           "--nproc_per_node", str(torch.cuda.device_count())] + cmd
else:
    cmd = [sys.executable] + cmd
print("running:", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

# --- final val ---------------------------------------------------------------
final_md = "/kaggle/working/final_val.md"
subprocess.run([sys.executable, "eval_val.py", "--ckpt", "/kaggle/working/out/ckpt.pt",
                "--val-bin", val_bin, "--batches", "20", "--batch-size", "8",
                "--device", "cuda", "--out", final_md], check=True)
final_val = val_loss_from_md(final_md)
ckpt = torch.load("/kaggle/working/out/ckpt.pt", map_location="cpu", weights_only=False)
manifest = {
    "experiment": "X64-v150-smoke",
    "code_commit": CODE_COMMIT,
    "params": params,
    "gpus": torch.cuda.device_count(),
    "max_minutes": MAX_MINUTES,
    "init_val": init_val,
    "init_gate": INIT_GATE,
    "final_val": final_val,
    "steps": int(ckpt.get("step", -1)),
    "tokens": int(ckpt.get("tokens", -1)),
    "init_ckpt": {"path": init_ckpt, "bytes": os.path.getsize(init_ckpt),
                  "sha256": sha256_file(init_ckpt)},
    "train_bin": {"path": train_bin, "bytes": os.path.getsize(train_bin),
                  "sha256": sha256_file(train_bin)},
    "val_bin": {"path": val_bin, "bytes": os.path.getsize(val_bin),
                "sha256": sha256_file(val_bin)},
}
Path("/kaggle/working/run_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
