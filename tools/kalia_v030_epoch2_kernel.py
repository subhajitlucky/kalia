"""Kaggle GPU kernel: v0.3.0 — the second epoch of v0.2.0.

Registered as ``docs/preregistrations/2026-10-27-v030-second-epoch.md``. This is
the first actual model-training run since v0.2.0: continue the step-4,770
checkpoint for 2,000 steps on the same 2.4B ``train.bin``, with a fresh cosine
(200-step warmup, peak 6e-4, floor 6e-5). No replay, no new data, no architecture
change.

The run is capped at 11 hours; a clean stop publishes the checkpoint and logs.
The primary val is a separate 100-batch eval on the final checkpoint, not a
training-log point.

Outputs: ``out/`` (ckpt.pt, train_log.csv, val_log.csv), ``v030-epoch2.yaml``,
and ``run_manifest.json`` recording the code fingerprint, every input hash, the
schedule and the budget.
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ARM = "epoch2"  # staging pins this
KERNEL_ID = "unknown"  # staging pins, from the committed metadata
KERNEL_REVISION = "unknown"  # staging pins, from git HEAD

STEPS = 2000
WARMUP = 200
MAX_MINUTES = 660
SEED = 1401
OVERRIDES = {
    "max_steps": STEPS,
    "update_budget": STEPS,
    "warmup_steps": WARMUP,
    "eval_interval": 100,
    "eval_steps": 100,
    "sample_interval": 0,
    "seed": SEED,
    "rewarm": True,
}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_fingerprint(root) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix in (".py", ".yaml"):
            h.update(str(p.relative_to(root)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def write_json(path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def write_run_manifest(out_path, **fields) -> dict:
    fields.update(
        kernel_id=KERNEL_ID,
        kernel_revision=KERNEL_REVISION,
        recorded=datetime.now(timezone.utc).isoformat(),
    )
    write_json(out_path, fields)
    print("run manifest: " + json.dumps(fields, sort_keys=True))
    return fields


# --- the code, resolved the same way every kernel resolves it ----------------
hits = sorted(glob.glob("/kaggle/input/**/train.py", recursive=True))
preferred = [p for p in hits if "/kalia-code-dev/" in p]
assert preferred, (
    "kalia-code-dev not attached or lacks train.py; other matches "
    "(do NOT trust them, they are stale bundles): " + str(hits)
)
code_root = os.path.dirname(preferred[0])
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("train.py", "configs/kalia-v020.yaml", "eval_val.py"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"
print("code:", code_root, "| arm:", ARM)

# --- the full corpus and the v0.2.0 checkpoint -------------------------------
train_hits = sorted(glob.glob("/kaggle/input/**/train.bin", recursive=True))
preferred_train = [p for p in train_hits if "/kalia-prep-v2b/" in p]
train_hits = preferred_train or train_hits
assert train_hits, "attach subhajitlucky/kalia-prep-v2b for train.bin"
data_dir = os.path.dirname(train_hits[0])
train_bin = train_hits[0]
val_bin = os.path.join(data_dir, "val.bin")
assert os.path.exists(val_bin), f"val.bin missing next to {train_bin}"
print("data:", data_dir)

ckpt_hits = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
preferred_ckpt = [p for p in ckpt_hits if "/kalia-v020-ckpt/" in p]
ckpt_hits = preferred_ckpt or ckpt_hits
assert ckpt_hits, "attach subhajitlucky/kalia-v020-ckpt"
resume = ckpt_hits[0]
print("resume:", resume)

# --- build the config from v0.2.0; only the registered keys move -------------
import yaml  # noqa: E402

base = yaml.safe_load(open("configs/kalia-v020.yaml"))
cfg = yaml.safe_load(open("configs/kalia-v020.yaml"))
cfg["train"].update(OVERRIDES)
assert cfg["model"] == base["model"], "the model block must be unchanged"
assert cfg["train"]["learning_rate"] == base["train"]["learning_rate"]
assert cfg["train"]["optimizer"] == "muon"
assert cfg["train"].get("lr_schedule", "cosine") == "cosine"
assert cfg["train"]["rewarm"] is True
assert cfg["train"]["max_steps"] == STEPS
assert cfg["train"]["update_budget"] == STEPS
cfg_path = "/kaggle/working/v030-epoch2.yaml"
yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
print("config:", cfg_path)
print("train overrides:", json.dumps(OVERRIDES, sort_keys=True))

write_run_manifest(
    "/kaggle/working/run_manifest.json",
    experiment="v030-epoch2",
    seed=SEED,
    steps=STEPS,
    warmup_steps=WARMUP,
    max_minutes=MAX_MINUTES,
    schedule="cosine (re-warm)",
    overrides=OVERRIDES,
    config_path=cfg_path,
    config_sha256=sha256_file(cfg_path),
    code_snapshot=code_root,
    code_fingerprint=code_fingerprint(code_root),
    data_dir=data_dir,
    train_bin={"path": train_bin, "bytes": os.path.getsize(train_bin),
               "sha256": sha256_file(train_bin)},
    val_bin={"path": val_bin, "bytes": os.path.getsize(val_bin),
             "sha256": sha256_file(val_bin)},
    resume_from={"path": resume, "bytes": os.path.getsize(resume),
                 "sha256": sha256_file(resume)},
    replay_bin=None,
)

# --- train -------------------------------------------------------------------
cmd = [
    "train.py",
    "--config", cfg_path,
    "--data-dir", data_dir,
    "--out-dir", "/kaggle/working/out",
    "--resume-from", resume,
    "--seed", str(SEED),
    "--max-minutes", str(MAX_MINUTES),
]
if torch.cuda.device_count() >= 2:
    cmd = [sys.executable, "-m", "torch.distributed.run",
           "--nproc_per_node", str(torch.cuda.device_count())] + cmd
else:
    cmd = [sys.executable] + cmd

print("running:", " ".join(cmd))
sys.exit(subprocess.run(cmd).returncode)
