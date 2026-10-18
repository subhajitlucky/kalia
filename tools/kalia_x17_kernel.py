"""Kaggle GPU kernel: X17 — does intra-document masking help?

Registered as ``docs/preregistrations/2026-09-28-X17-doc-mask.md``; design
amended by ``2026-10-18-X17-amendment-1.md`` (paired two-seed, DM-1b
concordance, DM-5 paired spread). This kernel runs the whole experiment: four
arms, 30M parameters, 500 steps, one machine.

| arm | config | doc_mask | seeds |
|---|---|---|---|
| control | ``micro-base`` | false | 1338, 1339 |
| treatment | ``micro-docmask`` | true | 1338, 1339 |

The two configs differ by exactly one line — ``doc_mask`` — which is the whole
design constraint, and a test asserts it.

Seed 1337 is **excluded**: it generated the hypotheses this project has been
testing, and X21's own registration excluded it as circular. Fresh seeds are
1338/1339, the pair D48 introduced.

**On the word "interleaved".** The four arms run back to back in one kernel,
control then treatment within each seed, so a machine that drifts loses both
arms of a pair equally. That is not the modded-nanogpt interleaving, which
alternates across many runs. X21's builder drew the same distinction rather than
glossing it, and this kernel makes the same claim and no stronger one.

**The defect this measures.** The corpus is EOS-delimited and windows are drawn
without document awareness, so most 512-token windows span two or three
documents and the model attends across every boundary. DM-4 measures that the
mask actually engages, on real batches, before any training happens — a fix whose
mask is inert would otherwise produce a null indistinguishable from "masking does
not help".

Outputs: ``out/<arm>-s<seed>/`` per arm (val_log.csv, ckpt.pt), the four configs
it built, and ``run_manifest.json`` recording the code fingerprint, every input
hash and the seed pair.
"""

import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ARM = "docmask"  # staging pins this; it names the experiment and its outputs
KERNEL_ID = "unknown"  # staging pins, from the committed metadata
KERNEL_REVISION = "unknown"  # staging pins, from git HEAD

SEEDS = ("1338", "1339")
STEPS = 500
# (arm label, base config). Ordered within each seed so the pair alternates.
PAIRS = (("control", "micro-base"), ("treatment", "micro-docmask"))
RUNS = tuple(
    (f"{label}-s{seed}", cfg, seed)
    for seed in SEEDS
    for label, cfg in PAIRS
)
CONTROL_LABEL, TREATMENT_LABEL = PAIRS[0][0], PAIRS[1][0]
DM1_BAR = -0.010  # unchanged from the original registration


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
    X29-v1). Two runs can now be shown to be code-identical instead of assumed.
    """
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix in (".py", ".yaml"):
            h.update(str(p.relative_to(root)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def write_run_manifest(out_path, **fields) -> dict:
    fields.update(
        kernel_id=KERNEL_ID,
        kernel_revision=KERNEL_REVISION,
        recorded=datetime.now(timezone.utc).isoformat(),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(fields, fh, indent=2, sort_keys=True)
    print("run manifest: " + json.dumps(fields, sort_keys=True))
    return fields


def mask_engagement(data_dir: str, windows: int = 32) -> float:
    """DM-4, measured on real batches before training.

    Mean fraction of causally-eligible positions the mask leaves attended. A
    working document mask must put this below 1.0; 1.0 means the mask did
    nothing and the arm is a duplicate of the control.
    """
    import torch

    from data import TokenDataset, document_mask

    ds = TokenDataset(os.path.join(data_dir, "train.bin"), 512)
    gen = torch.Generator().manual_seed(1234)
    attended = eligible = 0
    seen = 0
    while seen < windows:
        x, _ = ds.get_batch(8, torch.device("cpu"), gen)
        attn, _ = document_mask(x)
        b, t = x.shape
        attended += int(attn.sum())
        eligible += b * t * (t + 1) // 2
        seen += b
    return attended / eligible


# --- the code, resolved the same way every kernel resolves it ----------------
hits = sorted(glob.glob("/kaggle/input/**/ablate.py", recursive=True))
preferred = [p for p in hits if "/kalia-code-dev/" in p]
assert preferred, (
    "kalia-code-dev not attached or lacks ablate.py; other matches "
    "(do NOT trust them, they are stale bundles): " + str(hits)
)
code_root = os.path.dirname(preferred[0])
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("train.py", "ablate.py", "data.py",
            "configs/micro-base.yaml", "configs/micro-docmask.yaml"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"
print("code:", code_root)

data_hits = sorted(glob.glob("/kaggle/input/**/train.bin", recursive=True))
assert data_hits, "attach subhajitlucky/kalia-prep-v2b as a kernel source for train.bin"
data_dir = os.path.dirname(data_hits[0])
assert os.path.exists(os.path.join(data_dir, "val.bin")), "val.bin missing next to train.bin"
print("data:", data_dir)

# --- the one-line difference must actually be one line ----------------------
import yaml  # noqa: E402

base_cfg = yaml.safe_load(open("configs/micro-base.yaml"))
mask_cfg = yaml.safe_load(open("configs/micro-docmask.yaml"))
differing = {k for k in ("model", "train") if base_cfg[k] != mask_cfg[k]}
assert differing == {"train"}, f"the two arms differ in more than doc_mask: {differing}"
changed = {
    k
    for k in set(base_cfg["train"]) | set(mask_cfg["train"])
    if base_cfg["train"].get(k) != mask_cfg["train"].get(k)
}
assert changed == {"doc_mask"}, f"the treatment changed more than doc_mask: {changed}"
assert base_cfg["train"]["doc_mask"] is False and mask_cfg["train"]["doc_mask"] is True
print("configs differ in exactly one key: train.doc_mask")

# --- DM-4, before any GPU time ----------------------------------------------
engagement = mask_engagement(data_dir)
print(f"DM-4 mask engagement: {engagement:.4f} of causally-eligible positions attended")
assert engagement < 1.0, (
    f"the document mask does not engage ({engagement:.4f}); the treatment arm "
    "would be a duplicate of the control and DM-1 would be unreadable"
)

# --- build one config per (arm, seed) ---------------------------------------
# ablate.py resolves `configs/<arm>.yaml` then `<arm>.yaml` from the working
# directory, so per-(arm, seed) configs let the registered four-arm design run
# on the existing runner instead of a new training harness.
built = {}
for run, cfg_name, seed in RUNS:
    cfg = yaml.safe_load(open(f"configs/{cfg_name}.yaml"))
    cfg["train"]["seed"] = int(seed)
    yaml.safe_dump(cfg, open(f"{run}.yaml", "w"))
    built[run] = {"config": cfg_name, "seed": int(seed),
                  "doc_mask": cfg["train"]["doc_mask"]}
    print(f"built {run}.yaml from {cfg_name} (seed {seed}, doc_mask {cfg['train']['doc_mask']})")

write_run_manifest(
    "/kaggle/working/run_manifest.json",
    experiment=ARM,
    seeds=[int(s) for s in SEEDS],
    steps=STEPS,
    arms=built,
    dm1_bar=DM1_BAR,
    dm4_mask_engagement=round(engagement, 6),
    code_snapshot=code_root,
    code_fingerprint=code_fingerprint(code_root),
    data_dir=data_dir,
    train_bin={"path": os.path.join(data_dir, "train.bin"),
               "bytes": os.path.getsize(os.path.join(data_dir, "train.bin")),
               "sha256": sha256_file(os.path.join(data_dir, "train.bin"))},
    val_bin={"path": os.path.join(data_dir, "val.bin"),
             "bytes": os.path.getsize(os.path.join(data_dir, "val.bin")),
             "sha256": sha256_file(os.path.join(data_dir, "val.bin"))},
)

# --- run the four arms, paired by seed --------------------------------------
order = ",".join(run for run, _, _ in RUNS)
print(f"\n===== X17 ({ARM}) — arms in order: {order} =====")
proc = subprocess.run(
    [sys.executable, "ablate.py", "--arms", order, "--steps", str(STEPS),
     "--data-dir", data_dir],
)
if proc.returncode != 0:
    sys.exit(proc.returncode)

# --- the registered verdict, computed from the val logs ---------------------
def final_val(run: str) -> float | None:
    path = Path("/kaggle/working/out") / run / "val_log.csv"
    if not path.exists():
        return None
    rows = path.read_text().strip().splitlines()[1:]
    return float(rows[-1].split(",")[1]) if rows else None


def curve(run: str) -> dict:
    path = Path("/kaggle/working/out") / run / "val_log.csv"
    if not path.exists():
        return {}
    rows = path.read_text().strip().splitlines()[1:]
    return {int(r.split(",")[0]): float(r.split(",")[1]) for r in rows if r}


print("\n" + "=" * 64)
print("X17 REGISTERED VERDICT")
deltas = {}
for seed in SEEDS:
    c = final_val(f"{CONTROL_LABEL}-s{seed}")
    t = final_val(f"{TREATMENT_LABEL}-s{seed}")
    if c is None or t is None:
        print(f"  seed {seed}: an arm produced no val log — no verdict")
        continue
    d = t - c
    deltas[seed] = d
    print(f"  seed {seed}: control {c:.4f}  treatment {t:.4f}  delta {d:+.4f}")

if len(deltas) == len(SEEDS):
    mean = sum(deltas.values()) / len(deltas)
    spread = max(deltas.values()) - min(deltas.values())
    print(f"  mean paired delta {mean:+.4f} (DM-1 bar {DM1_BAR:+.3f})")
    print(f"  DM-1  {'PASS' if mean <= DM1_BAR else 'FAIL'}")
    print(f"  DM-1b {'PASS' if all(d < 0 for d in deltas.values()) else 'FAIL'}"
          "  (both deltas negative)")
    print(f"  DM-5  paired-difference spread {spread:.4f} — reported, no threshold")
    early = 0
    for seed in SEEDS:
        c, t = curve(f"{CONTROL_LABEL}-s{seed}"), curve(f"{TREATMENT_LABEL}-s{seed}")
        points = sorted(set(c) & set(t))
        if sum(1 for s in points if t[s] < c[s]) >= math.ceil(len(points) * 3 / 5):
            early += 1
    print(f"  DM-2  {'PASS' if early == len(SEEDS) else 'FAIL'}"
          f"  (treatment ahead at >=3/5 points at both seeds; {early}/{len(SEEDS)} seeds)")
    print(f"  DM-4  PASS  mask engagement {engagement:.4f} < 1.0")
print("Verdict is recorded by hand from these numbers and the archived logs, not")
print("asserted here: DM-3 (step-time ratio) needs the ablate output above.")
