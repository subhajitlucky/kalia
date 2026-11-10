"""Kaggle GPU kernel: X64 150M full continued pretraining.

One 7.5-hour session per run. The first session expands v0.3.1 to 38 layers and
starts from the zero-init checkpoint. Later sessions resume the latest
checkpoint from the private HF repo. Every session pushes its checkpoint and
logs to HF, so a killed Kaggle session loses at most the last 30 minutes.
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
CODE_COMMIT = "PLACEHOLDER"
HF_REPO = "kalia-lm/kalia-v150-run"
MAX_MINUTES = 450
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
for req in ("train.py", "model.py", "optim.py", "data.py", "eval_val.py"):
    assert os.path.exists(req), f"{req} missing from code snapshot"
print("code:", code_root, flush=True)

RAW = f"https://raw.githubusercontent.com/subhajitlucky/kalia/{CODE_COMMIT}"
for name in ("configs/kalia-v150-session.yaml", "tools/depth_expand.py"):
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{RAW}/{name}", name)
print("fetched X64 full config/expander", flush=True)

train_hits = sorted(glob.glob("/kaggle/input/**/train.bin", recursive=True))
preferred_train = [p for p in train_hits if "/kalia-prep-v2b/" in p]
train_hits = preferred_train or train_hits
assert train_hits, "attach subhajitlucky/kalia-prep-v2b for train.bin"
data_dir = os.path.dirname(train_hits[0])
train_bin = train_hits[0]
val_bin = os.path.join(data_dir, "val.bin")
assert os.path.exists(val_bin), f"val.bin missing next to {train_bin}"
print("data:", data_dir, flush=True)

base_hits = sorted(glob.glob("/kaggle/input/**/model.safetensors", recursive=True))
preferred_base = [p for p in base_hits if "/kalia-v031-ckpt/" in p]
base_hits = preferred_base or base_hits
assert base_hits, "attach subhajitlucky/kalia-v031-ckpt (model.safetensors + config.json)"
base_dir = os.path.dirname(base_hits[0])
assert os.path.exists(os.path.join(base_dir, "config.json")), base_dir
print("base checkpoint:", base_dir, flush=True)

HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "add HF_TOKEN to Kaggle Secrets; without it checkpoints do not survive sessions"

from huggingface_hub import HfApi, list_repo_files  # noqa: E402

api = HfApi(token=HF_TOKEN)
try:
    repo_files = list_repo_files(HF_REPO, token=HF_TOKEN)
except Exception as exc:  # repo may not exist yet
    print("resume check: repo not readable:", exc, flush=True)
    repo_files = []
has_ckpt = "checkpoints/ckpt.pt" in repo_files
has_started = "run_started.json" in repo_files
print("resume checkpoint present:", has_ckpt, flush=True)
if has_started and not has_ckpt:
    raise SystemExit(
        "run_started.json exists but checkpoints/ckpt.pt is missing; refusing "
        "to silently restart from scratch. Restore the checkpoint or remove "
        "run_started.json deliberately."
    )


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def val_loss_from_md(path):
    text = Path(path).read_text()
    m = re.search(r"Val loss:\s*([0-9.]+)", text)
    assert m, text
    return float(m.group(1))


init_ckpt = "/kaggle/working/kalia-v150-init.pt"
if has_ckpt:
    resume_arg = HF_REPO
    print("resuming from HF checkpoint", HF_REPO, flush=True)
else:
    subprocess.run([sys.executable, "tools/depth_expand.py",
                    "--base-dir", base_dir,
                    "--out", init_ckpt, "--layers", "38"], check=True)
    from model import GPT, GPTConfig  # noqa: E402
    payload = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    params = sum(p.numel() for p in GPT(GPTConfig(**payload["config"]["model"])).parameters())
    print("expanded params:", params, flush=True)

    init_md = "/kaggle/working/init_val.md"
    subprocess.run([sys.executable, "eval_val.py", "--ckpt", init_ckpt,
                    "--val-bin", val_bin, "--batches", "20", "--batch-size", "8",
                    "--device", "cuda", "--out", init_md], check=True)
    init_val = val_loss_from_md(init_md)
    print("init val", init_val, "gate", INIT_GATE, flush=True)
    assert init_val <= INIT_GATE, (
        f"expanded init val {init_val:.4f} exceeds gate {INIT_GATE:.4f}"
    )
    marker = {
        "experiment": "X64-v150-full",
        "code_commit": CODE_COMMIT,
        "init_val": init_val,
        "init_gate": INIT_GATE,
        "base_dir": base_dir,
        "started_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    Path("/kaggle/working/run_started.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n"
    )
    api.upload_file(
        path_or_fileobj="/kaggle/working/run_started.json",
        path_in_repo="run_started.json",
        repo_id=HF_REPO,
    )
    print("uploaded run_started.json", flush=True)
    resume_arg = init_ckpt

cmd = [sys.executable, "train.py",
       "--config", "configs/kalia-v150-session.yaml",
       "--data-dir", data_dir,
       "--out-dir", "/kaggle/working/out",
       "--resume-from", resume_arg,
       "--seed", "1501",
       "--max-minutes", str(MAX_MINUTES),
       "--hub-repo", HF_REPO]
if torch.cuda.device_count() >= 2:
    cmd = [sys.executable, "-m", "torch.distributed.run",
           "--nproc_per_node", str(torch.cuda.device_count())] + cmd
print("running:", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

session_md = "/kaggle/working/session_val.md"
subprocess.run([sys.executable, "eval_val.py",
                "--ckpt", "/kaggle/working/out/ckpt.pt",
                "--val-bin", val_bin, "--batches", "20", "--batch-size", "8",
                "--device", "cuda", "--out", session_md], check=True)
session_val = val_loss_from_md(session_md)
ckpt = torch.load("/kaggle/working/out/ckpt.pt", map_location="cpu", weights_only=False)
manifest = {
    "experiment": "X64-v150-full",
    "code_commit": CODE_COMMIT,
    "hf_repo": HF_REPO,
    "resumed_from_hf": bool(has_ckpt),
    "gpus": torch.cuda.device_count(),
    "max_minutes": MAX_MINUTES,
    "session_val": session_val,
    "steps": int(ckpt.get("step", -1)),
    "tokens": int(ckpt.get("tokens", -1)),
    "base_dir": base_dir,
    "train_bin": {"path": train_bin, "bytes": os.path.getsize(train_bin),
                  "sha256": sha256_file(train_bin)},
    "val_bin": {"path": val_bin, "bytes": os.path.getsize(val_bin),
                "sha256": sha256_file(val_bin)},
}
Path("/kaggle/working/session_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
api.upload_file(
    path_or_fileobj="/kaggle/working/session_manifest.json",
    path_in_repo="logs/session_manifest.json",
    repo_id=HF_REPO,
)
api.upload_file(
    path_or_fileobj=session_md,
    path_in_repo="logs/session_val.md",
    repo_id=HF_REPO,
)
print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
