"""Kaggle GPU kernel: X29 — patha replay (one arm per run).

``ARM`` in {uniform, krama, jata, ghana, noreplay}: a continual update of
v0.2.0, rewarm, 1000 update steps, replay at ``REPLAY`` of every batch — 0.10
for the patha ladder, 0.00 for the no-replay control. The replay shard is
carved in-runner from ``train.bin`` (10%, 32 blocks, seed 1337), the same shard
for every replay arm; only the presentation order differs. See
``docs/preregistrations/2026-10-11-X29-patha-replay.md``.

``noreplay`` exists because the 2026-10-18 CL-0 read could not attribute its
0.09-nat retention gap to replay: the no-replay arm it was read against (X28
static) differs in corpus and in passes over it as well. This arm runs the same
runner, the same corpus and the same step budget as ``uniform`` with the replay
fraction at zero, so replay is the only variable that moves.

``ARM``, ``SEED``, ``REPLAY``, ``KERNEL_ID`` and ``KERNEL_REVISION`` are patched
by ``tools/stage_x29_arm.py``. Separate kernels per arm: a failed run publishes
no outputs.

Each run writes ``out/x29-<arm>/run_manifest.json``: code fingerprint, every
input's sha256, the seed. The 2026-10-17 review's finding (4) — the seed-1402
replicas could not say which seed they ran or which code bundle they ran it
from, and both had to be established by re-running the stager.
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

ARM = "uniform"  # uniform | krama | jata | ghana | noreplay
SEED = "1401"  # staging patches this for the seed-1402 replication
REPLAY = "0.10"  # staging pins this; the no-replay control is "0.00"
KERNEL_ID = "unknown"  # staging patches, from the committed metadata
KERNEL_REVISION = "unknown"  # staging patches, from git HEAD

PATTERNS = ("uniform", "krama", "jata", "ghana")
replay_prob = float(REPLAY)
replay_pattern = ARM if ARM in PATTERNS else "uniform"
assert not (replay_prob > 0 and ARM == "noreplay"), f"{ARM} cannot carry a replay fraction"
assert not (replay_prob == 0 and ARM in PATTERNS), (
    f"{ARM} is a replay arm; the no-replay control is ARM='noreplay'"
)


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
hits = sorted(glob.glob("/kaggle/input/**/patha.py", recursive=True))
preferred = [p for p in hits if "/kalia-code-dev/" in p]
assert preferred, (
    "kalia-code-dev not attached or lacks patha.py; other matches "
    "(do NOT trust them, they are stale bundles): " + str(hits)
)
code_root = os.path.dirname(preferred[0])
if not os.path.exists(work):
    shutil.copytree(code_root, work)
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
# across the replay arms (fraction, blocks, seed fixed). The no-replay control
# does not carve it and is never handed the path, so it cannot pick one up.
replay_dir = "/kaggle/working/replay"
replay_bin = None
if replay_prob > 0:
    if not os.path.exists(f"{replay_dir}/replay.bin"):
        r = subprocess.run(
            [sys.executable, "make_replay_shard.py",
             "--train-bin", os.path.join(data_dir, "train.bin"),
             "--out-dir", replay_dir, "--fraction", "0.10",
             "--blocks", "32", "--seed", "1337"],
            text=True,
        )
        assert r.returncode == 0, "replay shard carve failed"
    replay_bin = f"{replay_dir}/replay.bin"

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
        "replay_prob": replay_prob,
        "replay_pattern": replay_pattern,
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
]
if replay_bin:
    cmd += ["--replay-bin", replay_bin]
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
    replay_prob=replay_prob,
    replay_pattern=replay_pattern,
    update_steps=1000,
    code_snapshot=code_root,
    code_fingerprint=code_fingerprint(code_root),
    data_dir=data_dir,
    train_bin=_input(os.path.join(data_dir, "train.bin")),
    val_bin=_input(os.path.join(data_dir, "val.bin")),
    resume_from=_input(resume_src),
    replay_bin=_input(replay_bin) if replay_bin else None,
)

print(f"\n===== x29-{ARM} (replay {replay_prob:.0%}, pattern {replay_pattern}) =====")
sys.exit(subprocess.run(cmd).returncode)
