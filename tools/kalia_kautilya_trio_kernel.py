"""Kaggle GPU kernel: X28 — the Kautilya trio (static | level | trend).

One arm per kernel run; ``ARM`` is set below, and each arm resumes from v0.2.0
at step 4770, seed 1401, `rewarm: true`, 1000 update steps. See
``docs/preregistrations/2026-10-09-X28-kautilya-trend.md``.

Separate kernels per arm on purpose: a failed run publishes no outputs, and the
X26 pilot lost a completed arm that way.
"""

import glob
import json
import os
import shutil
import subprocess
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ARM = "static"  # static | level | trend

ARM_SETTINGS = {
    "static": {"adaptive_every": 0, "adaptive_trend_window": 1},
    "level": {"adaptive_every": 100, "adaptive_trend_window": 1},
    "trend": {"adaptive_every": 100, "adaptive_trend_window": 3},
}

work = "/kaggle/working/kalia"
if not os.path.exists(work):
    hits = sorted(glob.glob("/kaggle/input/**/mixture.py", recursive=True))
    preferred = [p for p in hits if "/kalia-code-dev/" in p]
    assert preferred, (
        "kalia-code-dev not attached or lacks mixture.py; other matches "
        "(do NOT trust them, they are stale bundles): " + str(hits)
    )
    shutil.copytree(os.path.dirname(preferred[0]), work)
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
config["train"]["seed"] = 1401
yaml.safe_dump(config, open(cfg_path, "w"))

cmd = [
    "train.py",
    "--config", cfg_path,
    "--data-dir", data_dir,
    "--out-dir", out_dir,
    "--resume-from", resume_src,
    "--seed", "1401",
    "--sources", *NAMES,
    "--source-bins", *[shard_paths[n] for n in NAMES],
    "--source-weights", json.dumps(WEIGHTS),
]
if torch.cuda.device_count() >= 2:
    cmd = [sys.executable, "-m", "torch.distributed.run",
           "--nproc_per_node", str(torch.cuda.device_count())] + cmd
else:
    cmd = [sys.executable] + cmd
print(f"\n===== kautilya-{ARM} =====")
sys.exit(subprocess.run(cmd).returncode)
