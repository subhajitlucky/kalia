"""Kaggle GPU kernel: X35 — does the static source mixture move the 58M update?

Registered as ``docs/preregistrations/2026-10-26-X35-static-mixture-screen.md``.
The stager pins ``ARM`` to ``mix-screen`` or ``mix-replicate``; the stage decides
the seed and the arms, so a staged copy cannot silently run the wrong stage.

The primary metric is the last logged val point — step 5,200 with the base
config's ``eval_interval: 100``. The final step 5,270 is not an eval boundary;
Amendment 1 fixes this before any run.

Stage 1 (``mix-screen``, seed 1401) runs three static mixtures over the four
published 498M per-source shards, 500 update steps from v0.2.0:

| arm | FineWeb | TinyStories | Cosmopedia | Python |
|---|---|---|---|---|
| control | 0.60 | 0.20 | 0.15 | 0.05 |
| code10 | 0.55 | 0.20 | 0.15 | 0.10 |
| code25 | 0.40 | 0.20 | 0.15 | 0.25 |

Stage 2 (``mix-replicate``, seed 1402) runs control and code10 only, and refuses
to run unless the screen wrote ``advance: true`` (S-1 passed).

Only the source weights and the seed move. The model, schedule, token budget,
shards and validation set are fixed. No replay, matching the X28 static arm.

Outputs: ``out/<arm>-s<seed>/`` per arm (val_log.csv, train_log.csv,
per_source_log.csv, ckpt.pt), ``x35_screen.json`` or ``x35_verdict.json``, the
configs it built, and ``run_manifest.json`` recording the code fingerprint,
every input hash, the weights and the registered bars.

CL-0 is a separate CPU probe on the Stage 2 checkpoints, as the registration
says; this kernel does not spend GPU time on it.
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

ARM = "mix-screen"  # staging pins this: mix-screen | mix-replicate
KERNEL_ID = "unknown"  # staging pins, from the committed metadata
KERNEL_REVISION = "unknown"  # staging pins, from git HEAD

STEPS = 500
S1_BAR = -0.010
S3_COLLAPSE = 0.050
NAMES = ("fineweb", "tinystories", "cosmopedia", "python")
BASE_CONFIG = "configs/v030-rewarm-s1401.yaml"

CONTROL = {"fineweb": 0.60, "tinystories": 0.20, "cosmopedia": 0.15, "python": 0.05}
CODE10 = {"fineweb": 0.55, "tinystories": 0.20, "cosmopedia": 0.15, "python": 0.10}
CODE25 = {"fineweb": 0.40, "tinystories": 0.20, "cosmopedia": 0.15, "python": 0.25}

STAGES = {
    "mix-screen": {
        "seed": 1401,
        "arms": {"control": CONTROL, "code10": CODE10, "code25": CODE25},
    },
    "mix-replicate": {
        "seed": 1402,
        "arms": {"control": CONTROL, "code10": CODE10},
    },
}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_fingerprint(root) -> str:
    """Fingerprint the code the run actually imports, not the repo it came from."""
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


assert ARM in STAGES, f"ARM {ARM!r} is not one of {sorted(STAGES)}"
STAGE = STAGES[ARM]
SEED = STAGE["seed"]
WEIGHTS = STAGE["arms"]

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
for req in ("train.py", "mixture.py", "configs/kalia-v020.yaml", BASE_CONFIG,
            "tools/forgetting_probe.py"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"
print("code:", code_root, "| arm:", ARM, "| seed:", SEED)

# --- the data: published shards, canonical val, v0.2.0 checkpoint ------------
import torch  # noqa: E402
import yaml  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "kalia-lm/kalia-v020"
shard_paths = {}
for name in NAMES:
    shard_paths[name] = hf_hub_download(REPO, f"corpus/{name}.bin", repo_type="model")
    print(f"shard {name}: {os.path.getsize(shard_paths[name]) / 1e6:.0f} MB")

val_hits = sorted(glob.glob("/kaggle/input/**/val.bin", recursive=True))
preferred_val = [p for p in val_hits if "/kalia-prep-v2b/" in p]
val_hits = preferred_val or val_hits
assert val_hits, "attach subhajitlucky/kalia-prep-v2b as a kernel source for val.bin"
data_dir = os.path.dirname(val_hits[0])
print("val shard:", val_hits[0])

ckpt_hits = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
preferred_ckpt = [p for p in ckpt_hits if "/kalia-v020-ckpt/" in p]
ckpt_hits = preferred_ckpt or ckpt_hits
assert ckpt_hits, "attach subhajitlucky/kalia-v020-ckpt"
resume = ckpt_hits[0]
print("resume checkpoint:", resume)

# --- the config is fixed; only the seed moves -------------------------------
base = yaml.safe_load(open(BASE_CONFIG))
v020 = yaml.safe_load(open("configs/kalia-v020.yaml"))
assert base["train"]["max_steps"] == STEPS, base["train"]["max_steps"]
assert base["train"]["update_budget"] == STEPS, base["train"]["update_budget"]
assert base["train"].get("rewarm") is True, "the screen is a re-warm update"
assert base["model"] == v020["model"], "the model block must be unchanged"
assert base["train"].get("replay_prob", 0.0) == 0.0, "the screen has no replay"

for arm, weights in WEIGHTS.items():
    assert set(weights) == set(NAMES), arm
    assert abs(sum(weights.values()) - 1.0) < 1e-9, (arm, sum(weights.values()))

# --- Stage 2 gate: the screen must have advanced ----------------------------
screen_summary = None
if ARM == "mix-replicate":
    screen_hits = sorted(glob.glob("/kaggle/input/**/x35_screen.json", recursive=True))
    assert screen_hits, (
        "x35_screen.json not found; attach subhajitlucky/kalia-x35-mix-screen "
        "as a kernel source"
    )
    screen_summary = json.load(open(screen_hits[0]))
    assert screen_summary.get("advance") is True, (
        "Stage 1 did not advance (S-1 failed or was not read). Do not run the "
        "replication; the registration says stop after the screen."
    )
    print("screen summary:", json.dumps(screen_summary, sort_keys=True))

# --- run the arms ------------------------------------------------------------
def run_arm(arm: str, weights: dict) -> int:
    cfg = yaml.safe_load(open(BASE_CONFIG))
    cfg["train"]["seed"] = SEED
    cfg_path = f"/kaggle/working/x35-{arm}-s{SEED}.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
    out_dir = f"/kaggle/working/out/{arm}-s{SEED}"
    cmd = [
        "train.py",
        "--config", cfg_path,
        "--data-dir", data_dir,
        "--out-dir", out_dir,
        "--resume-from", resume,
        "--seed", str(SEED),
        "--sources", *NAMES,
        "--source-bins", *[shard_paths[n] for n in NAMES],
        "--source-weights", json.dumps(weights),
    ]
    if torch.cuda.device_count() >= 2:
        cmd = [sys.executable, "-m", "torch.distributed.run",
               "--nproc_per_node", str(torch.cuda.device_count())] + cmd
    else:
        cmd = [sys.executable] + cmd
    print(f"\n===== {arm} @ seed {SEED} — {weights} =====")
    return subprocess.run(cmd).returncode


def _input(path):
    return {"path": path, "bytes": os.path.getsize(path), "sha256": sha256_file(path)}


write_run_manifest(
    "/kaggle/working/run_manifest.json",
    experiment=ARM,
    seed=SEED,
    steps=STEPS,
    arms=WEIGHTS,
    base_config=BASE_CONFIG,
    base_config_sha256=sha256_file(BASE_CONFIG),
    code_snapshot=code_root,
    code_fingerprint=code_fingerprint(code_root),
    data_dir=data_dir,
    val_bin=_input(val_hits[0]),
    source_bins={n: _input(shard_paths[n]) for n in NAMES},
    resume_from=_input(resume),
    replay_bin=None,
    bars={"S1": S1_BAR, "S3_collapse": S3_COLLAPSE},
)

for arm, weights in WEIGHTS.items():
    rc = run_arm(arm, weights)
    if rc != 0:
        sys.exit(rc)


# --- read the val logs and compute the registered verdict --------------------
def final_val(arm: str) -> float | None:
    path = Path("/kaggle/working/out") / f"{arm}-s{SEED}" / "val_log.csv"
    if not path.exists():
        return None
    rows = path.read_text().strip().splitlines()[1:]
    return float(rows[-1].split(",")[1]) if rows else None


vals = {arm: final_val(arm) for arm in WEIGHTS}
missing = [a for a, v in vals.items() if v is None]
if missing:
    print(f"NO VERDICT: no val log for {missing}")
    write_json("/kaggle/working/x35_verdict.json",
               {"experiment": ARM, "seed": SEED, "vals": vals,
                "error": f"missing val logs: {missing}"})
    sys.exit(1)

print("\n" + "=" * 72)
print(f"X35 {ARM} — seed {SEED}, {STEPS} steps")
for arm, v in vals.items():
    print(f"  {arm:8s} {v:.4f}  {WEIGHTS[arm]}")

if ARM == "mix-screen":
    d10 = vals["code10"] - vals["control"]
    d25 = vals["code25"] - vals["control"]
    s1 = d10 <= S1_BAR
    s3 = d25 <= S3_COLLAPSE  # False means the dose-response arm collapsed
    advance = bool(s1)
    summary = {
        "experiment": "x35-screen",
        "seed": SEED,
        "steps": STEPS,
        "vals": vals,
        "deltas": {"code10": d10, "code25": d25},
        "bars": {"S1": S1_BAR, "S3_collapse": S3_COLLAPSE},
        "S1": bool(s1),
        "S3_within_boundary": bool(s3),
        "advance": advance,
        "weights": WEIGHTS,
        "run_manifest": "/kaggle/working/run_manifest.json",
    }
    print(f"  code10 - control: {d10:+.4f}  (S-1 bar {S1_BAR:+.3f}) "
          f"{'PASS' if s1 else 'FAIL'}")
    print(f"  code25 - control: {d25:+.4f}  (collapse boundary {S3_COLLAPSE:+.3f}) "
          f"{'within' if s3 else 'COLLAPSED'}")
    print(f"  advance to Stage 2: {'YES' if advance else 'NO'}")
    write_json("/kaggle/working/x35_screen.json", summary)
    print("screen summary written to /kaggle/working/x35_screen.json")
else:
    screen = screen_summary or {}
    prev = screen.get("vals", {})
    if not prev:
        print("NO VERDICT: screen summary has no vals")
        sys.exit(1)
    d10_1402 = vals["code10"] - vals["control"]
    d10_1401 = screen["deltas"]["code10"]
    mean = (d10_1401 + d10_1402) / 2
    same_sign = (d10_1401 < 0) == (d10_1402 < 0)
    s2 = d10_1402 <= S1_BAR and mean <= S1_BAR and same_sign
    verdict = {
        "experiment": "x35-replicate",
        "seeds": [1401, 1402],
        "steps": STEPS,
        "vals": {"1401": prev, "1402": vals},
        "deltas": {"1401": d10_1401, "1402": d10_1402},
        "mean": mean,
        "bars": {"S2": S1_BAR},
        "S2": bool(s2),
        "same_sign": bool(same_sign),
        "weights": WEIGHTS,
        "run_manifest": "/kaggle/working/run_manifest.json",
    }
    print(f"  code10 - control 1401: {d10_1401:+.4f}")
    print(f"  code10 - control 1402: {d10_1402:+.4f}")
    print(f"  two-seed mean: {mean:+.4f}  same sign: {same_sign}")
    print(f"  S-2 {'PASS' if s2 else 'FAIL'} (bar {S1_BAR:+.3f})")
    write_json("/kaggle/working/x35_verdict.json", verdict)
    print("verdict written to /kaggle/working/x35_verdict.json")
