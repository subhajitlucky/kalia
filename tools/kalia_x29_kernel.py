"""Kaggle GPU kernel: X29 — patha replay (one arm per run).

``ARM`` in {uniform, krama, jata, ghana}: a continual update of v0.2.0, seed
1401, rewarm, 1000 update steps, replay fixed at 10% of every batch. The
replay shard is carved in-runner from ``train.bin`` (10%, 32 blocks, seed
1337) — the same shard for every arm; only the presentation order differs.
See ``docs/preregistrations/2026-10-11-X29-patha-replay.md``.

``ARM`` and ``SEED`` are patched by ``tools/stage_x29_arm.py`` (a seed other
than 1401 is the registered second-seed replication). Separate kernels per arm:
a failed run publishes no outputs.
"""

import glob
import os
import shutil
import subprocess
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ARM = "uniform"  # uniform | krama | jata | ghana
SEED = "1401"  # staging patches this for the seed-1402 replication

work = "/kaggle/working/kalia"
if not os.path.exists(work):
    hits = sorted(glob.glob("/kaggle/input/**/patha.py", recursive=True))
    preferred = [p for p in hits if "/kalia-code-dev/" in p]
    assert preferred, (
        "kalia-code-dev not attached or lacks patha.py; other matches "
        "(do NOT trust them, they are stale bundles): " + str(hits)
    )
    shutil.copytree(os.path.dirname(preferred[0]), work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("train.py", "patha.py", "make_replay_shard.py", "configs/kalia-v020.yaml"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"

import torch  # noqa: E402
import yaml  # noqa: E402

data_hits = sorted(glob.glob("/kaggle/input/**/train.bin", recursive=True))
assert data_hits, "attach kalia-prep-v2b as a kernel source for train.bin"
data_dir = os.path.dirname(data_hits[0])
val_ok = os.path.exists(os.path.join(data_dir, "val.bin"))
assert val_ok, "val.bin missing next to train.bin"
print("data:", data_dir, "| gpus:", torch.cuda.device_count())

ckpt_hits = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
assert ckpt_hits, "attach subhajitlucky/kalia-v020-ckpt"
resume_src = ckpt_hits[0]

# The frozen replay shard: carved once per run from the corpus, identical
# across arms (fraction, blocks, seed fixed).
replay_dir = "/kaggle/working/replay"
if not os.path.exists(f"{replay_dir}/replay.bin"):
    r = subprocess.run(
        [sys.executable, "make_replay_shard.py",
         "--train-bin", os.path.join(data_dir, "train.bin"),
         "--out-dir", replay_dir, "--fraction", "0.10",
         "--blocks", "32", "--seed", "1337"],
        text=True,
    )
    assert r.returncode == 0, "replay shard carve failed"

config = yaml.safe_load(open("configs/kalia-v020.yaml"))
config["train"].update(
    {
        "max_steps": 1000,
        "update_budget": 1000,
        "warmup_steps": 100,
        "eval_interval": 100,
        "eval_steps": 20,
        "sample_interval": 0,
        "checkpoint_interval_minutes": 20,
        "rewarm": True,
        "replay_prob": 0.10,
        "replay_pattern": ARM,
        "seed": int(SEED),
    }
)
cfg_path = f"/kaggle/working/x29-{ARM}.yaml"
yaml.safe_dump(config, open(cfg_path, "w"))

out_dir = f"/kaggle/working/out/x29-{ARM}"
cmd = [
    "train.py",
    "--config", cfg_path,
    "--data-dir", data_dir,
    "--out-dir", out_dir,
    "--resume-from", resume_src,
    "--seed", SEED,
    "--replay-bin", f"{replay_dir}/replay.bin",
]
if torch.cuda.device_count() >= 2:
    cmd = [sys.executable, "-m", "torch.distributed.run",
           "--nproc_per_node", str(torch.cuda.device_count())] + cmd
else:
    cmd = [sys.executable] + cmd
print(f"\n===== x29-{ARM} (replay pattern: {ARM}) =====")
sys.exit(subprocess.run(cmd).returncode)
