"""Kaggle GPU kernel: X28 — the Kautilya trio (static | level | trend).

One arm per kernel run; ``ARM`` is set below, and each arm resumes from v0.2.0
at step 4770, seed 1401, `rewarm: true`, 1000 update steps. See
``docs/preregistrations/2026-10-09-X28-kautilya-trend.md``.

Note what this arm *is*, because the 2026-10-18 CL-0 read turned on it: it
samples the four published per-source shards directly at their corpus weights
(60/20/15/5) and passes no ``--replay-bin``, so `replay_prob` stays at its 0.0
default — no replay, from 498,002,764 tokens, which the 1000-step budget
(~524M tokens) covers about once. The X29 ``uniform`` arm it was compared
against samples the 2.4B ``train.bin`` about 0.2 times and adds 10% replay. Two
variables, not one; neither direction is attributable to replay alone.

``ARM``, ``SEED``, ``KERNEL_ID`` and ``KERNEL_REVISION`` are patched by
``tools/stage_x29_arm.py`` (a seed other than 1401 is the registered second-seed
replication). Separate kernels per arm on purpose: a failed run publishes no
outputs, and the X26 pilot lost a completed arm that way.

Each run writes ``out/kautilya-<arm>/run_manifest.json``: code fingerprint,
every input's sha256, the seed — the 2026-10-17 review's finding (4).
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ARM = "static"  # static | level | trend
SEED = "1401"  # staging patches this for the seed-1402 replication
KERNEL_ID = "unknown"  # staging patches, from the committed metadata
KERNEL_REVISION = "unknown"  # staging patches, from git HEAD

ARM_SETTINGS = {
    "static": {"adaptive_every": 0, "adaptive_trend_window": 1},
    "level": {"adaptive_every": 100, "adaptive_trend_window": 1},
    "trend": {"adaptive_every": 100, "adaptive_trend_window": 3},
}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_fingerprint(root) -> str:
    """Fingerprint the code the run actually imports, not the repo it came from.

    Kernels mount the ``kalia-code-dev`` snapshot, which lags HEAD, and a stale
    bundle has silently changed what an arm meant three times (I10, I23,
    X29-v1). Recording the snapshot's own fingerprint is what makes two runs
    comparable on code rather than on the claim that they were.
    """
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix in (".py", ".yaml"):
            h.update(str(p.relative_to(root)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def write_run_manifest(out_dir, **fields) -> dict:
    fields.update(
        kernel_id=KERNEL_ID,
        kernel_revision=KERNEL_REVISION,
        recorded=datetime.now(timezone.utc).isoformat(),
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_manifest.json"), "w") as fh:
        json.dump(fields, fh, indent=2, sort_keys=True)
    print("run manifest: " + json.dumps(fields, sort_keys=True))
    return fields


work = "/kaggle/working/kalia"
hits = sorted(glob.glob("/kaggle/input/**/mixture.py", recursive=True))
preferred = [p for p in hits if "/kalia-code-dev/" in p]
assert preferred, (
    "kalia-code-dev not attached or lacks mixture.py; other matches "
    "(do NOT trust them, they are stale bundles): " + str(hits)
)
code_root = os.path.dirname(preferred[0])
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("train.py", "mixture.py", "configs/kalia-kautilya.yaml", "configs/kalia-v020.yaml"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"

import torch  # noqa: E402
import yaml  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "kalia-lm/kalia-v020"
print("arm:", ARM, "| gpus:", torch.cuda.device_count())

shard_paths = {}
for name in ("fineweb", "tinystories", "cosmopedia", "python"):
    shard_paths[name] = hf_hub_download(REPO, f"corpus/{name}.bin", repo_type="model")
    print(f"shard {name}: {os.path.getsize(shard_paths[name]) / 1e6:.0f} MB")

val_hits = sorted(glob.glob("/kaggle/input/**/val.bin", recursive=True))
assert val_hits, "attach kalia-prep-v2b as a kernel source for the canonical val.bin"
data_dir = os.path.dirname(val_hits[0])
print("val shard:", val_hits[0])

ckpt_hits = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
resume_src = ckpt_hits[0] if ckpt_hits else REPO
print("resume from:", resume_src)

WEIGHTS = {"fineweb": 0.6, "tinystories": 0.2, "cosmopedia": 0.15, "python": 0.05}
NAMES = list(WEIGHTS)

out_dir = f"/kaggle/working/out/kautilya-{ARM}"
cfg_path = f"/kaggle/working/kautilya-{ARM}.yaml"
config = yaml.safe_load(open("configs/kalia-kautilya.yaml"))
config["train"].update(ARM_SETTINGS[ARM])
config["train"]["rewarm"] = True
config["train"]["seed"] = int(SEED)
yaml.safe_dump(config, open(cfg_path, "w"))

cmd = [
    "train.py",
    "--config", cfg_path,
    "--data-dir", data_dir,
    "--out-dir", out_dir,
    "--resume-from", resume_src,
    "--seed", SEED,
    "--sources", *NAMES,
    "--source-bins", *[shard_paths[n] for n in NAMES],
    "--source-weights", json.dumps(WEIGHTS),
]
if torch.cuda.device_count() >= 2:
    cmd = [sys.executable, "-m", "torch.distributed.run",
           "--nproc_per_node", str(torch.cuda.device_count())] + cmd
else:
    cmd = [sys.executable] + cmd

# Written before training starts, so a run that dies still leaves its record.
def _input(path):
    return {"path": path, "bytes": os.path.getsize(path), "sha256": sha256_file(path)}


write_run_manifest(
    out_dir,
    arm=ARM,
    seed=int(SEED),
    max_steps=config["train"]["max_steps"],
    update_budget=config["train"].get("update_budget"),
    adaptive_every=config["train"]["adaptive_every"],
    adaptive_trend_window=config["train"]["adaptive_trend_window"],
    source_weights=WEIGHTS,
    source_order=NAMES,
    code_snapshot=code_root,
    code_fingerprint=code_fingerprint(code_root),
    data_dir=data_dir,
    val_bin=_input(os.path.join(data_dir, "val.bin")),
    source_bins={n: _input(shard_paths[n]) for n in NAMES},
    resume_from=_input(resume_src) if os.path.exists(resume_src) else resume_src,
    replay_bin=None,
)

print(f"\n===== kautilya-{ARM} =====")
sys.exit(subprocess.run(cmd).returncode)
