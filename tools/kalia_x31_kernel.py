"""Kaggle GPU kernel: X31 — does QK-Norm + logit soft-cap earn its place?

Registered as ``docs/preregistrations/2026-10-21-X31-qk-softcap-isolation.md``.
This kernel runs the whole experiment: four conditions at two fresh seeds,
30M parameters, 763 steps, one machine.

| condition | qk_norm | logit_softcap | seeds |
|---|---|---|---|
| control | false | 0.0 | 1338, 1339 |
| qk | true | 0.0 | 1338, 1339 |
| cap | false | 30.0 | 1338, 1339 |
| both | true | 30.0 | 1338, 1339 |

The shipped recipe's claim is the *bundle*: D8 credited QK-Norm + soft-cap with
a further -0.0834 nats on one seed, and neither component was ever isolated.
The four conditions turn that one bundled number into a 2x2, and the two fresh
seeds turn one observation into a paired test.

Seed 1337 is **excluded**: it generated the hypothesis, and including it would
be circular. Seeds 1338/1339 are the pair D48 introduced; X21 measured the
control's paired spread on them at 0.0139.

**Sequential, not interleaved.** Conditions run control -> qk -> cap -> both
within each seed, then the next seed. They share a machine and a session, but
do not alternate across many runs the way the modded-nanogpt records do. The
write-up must say that rather than imply a stronger design.

Outputs: ``out/<condition>-s<seed>/`` per arm (val_log.csv, ckpt.pt), the eight
configs it built, ``x31_verdict.json``, and ``run_manifest.json`` recording the
code fingerprint, every input hash and the registered bars.
"""

import copy
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

ARM = "qk-isolation"  # staging pins this; it names the experiment and its outputs
KERNEL_ID = "unknown"  # staging pins, from the committed metadata
KERNEL_REVISION = "unknown"  # staging pins, from git HEAD

SEEDS = ("1338", "1339")
STEPS = 763
# (label, committed config, qk_norm, logit_softcap). Ordered within each seed
# so a machine that drifts loses all four arms of a pair equally.
CONDITIONS = (
    ("control", "micro-muon", False, 0.0),
    ("qk", "micro-qk", True, 0.0),
    ("cap", "micro-cap", False, 30.0),
    ("both", "micro-qkcap", True, 30.0),
)
RUNS = tuple(
    (f"{label}-s{seed}", cfg, seed, qk, cap)
    for seed in SEEDS
    for label, cfg, qk, cap in CONDITIONS
)
Q1_BAR = -0.020
Q2_BAR = -0.010
Q3_MARGIN = 0.010


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
for req in ("train.py", "ablate.py", "data.py", "configs/micro-muon.yaml"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"
print("code:", code_root)

data_hits = sorted(glob.glob("/kaggle/input/**/train.bin", recursive=True))
assert data_hits, "attach subhajitlucky/kalia-prep-v2b as a kernel source for train.bin"
preferred_data = [p for p in data_hits if "/kalia-prep-v2b/" in p]
data_hits = preferred_data or data_hits
data_dir = os.path.dirname(data_hits[0])
assert os.path.exists(os.path.join(data_dir, "val.bin")), "val.bin missing next to train.bin"
print("data:", data_dir)

# --- build the 2x2 from the control, and prove nothing else moved -----------
import yaml  # noqa: E402

base = yaml.safe_load(open("configs/micro-muon.yaml"))
assert base["train"]["max_steps"] == STEPS, (
    f"control max_steps {base['train']['max_steps']} != registered {STEPS}"
)
assert "qk_norm" not in base["model"] and "logit_softcap" not in base["model"], (
    "the control config already carries a QK/soft-cap flag; the 2x2 would not be "
    "a 2x2"
)

artifact_dir = Path("/kaggle/working/x31-configs")
artifact_dir.mkdir(parents=True, exist_ok=True)
built = {}
for run, cfg_name, seed, qk, cap in RUNS:
    cfg = copy.deepcopy(base)
    cfg["model"]["qk_norm"] = qk
    cfg["model"]["logit_softcap"] = cap
    cfg["train"]["seed"] = int(seed)
    # Write into configs/ so ablate.py's first resolution path finds exactly
    # this file. The staged snapshot may or may not carry a newer copy.
    path = Path("configs") / f"{run}.yaml"
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    shutil.copy2(path, artifact_dir / path.name)
    built[run] = {
        "config": cfg_name,
        "seed": int(seed),
        "qk_norm": qk,
        "logit_softcap": cap,
        "sha256": sha256_file(path),
    }
    print(f"built {run}.yaml from {cfg_name} "
          f"(seed {seed}, qk_norm {qk}, logit_softcap {cap})")

# The registered design is a true 2x2 and nothing else. Re-read the files
# rather than trusting the loop above.
for run, cfg_name, seed, qk, cap in RUNS:
    cfg = yaml.safe_load(open(f"configs/{run}.yaml"))
    assert cfg["model"] == {**base["model"], "qk_norm": qk, "logit_softcap": cap}, run
    assert cfg["train"] == {**base["train"], "seed": int(seed)}, run
print("config invariant: only qk_norm, logit_softcap and seed move")

write_run_manifest(
    "/kaggle/working/run_manifest.json",
    experiment=ARM,
    seeds=[int(s) for s in SEEDS],
    steps=STEPS,
    arms=built,
    bars={"q1": Q1_BAR, "q2": Q2_BAR, "q3_margin": Q3_MARGIN},
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

# --- run the eight arms, paired by seed --------------------------------------
order = ",".join(run for run, _, _, _, _ in RUNS)
print(f"\n===== X31 ({ARM}) — arms in order: {order} =====")
proc = subprocess.run(
    [sys.executable, "ablate.py", "--arms", order, "--steps", str(STEPS),
     "--data-dir", data_dir, "--out-root", "/kaggle/working/out"],
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


deltas = {}
by_seed = {}
print("\n" + "=" * 72)
print("X31 REGISTERED VERDICT")
print(f"{'seed':>6s} {'control':>9s} {'qk':>9s} {'cap':>9s} {'both':>9s} "
      f"{'d_qk':>9s} {'d_cap':>9s} {'d_both':>9s}")
for seed in SEEDS:
    vals = {label: final_val(f"{label}-s{seed}") for label in
            ("control", "qk", "cap", "both")}
    if any(v is None for v in vals.values()):
        print(f"  seed {seed}: an arm produced no val log — no verdict")
        continue
    by_seed[seed] = vals
    d = {label: vals[label] - vals["control"] for label in ("qk", "cap", "both")}
    deltas[seed] = d
    print(f"{seed:>6s} {vals['control']:9.4f} {vals['qk']:9.4f} "
          f"{vals['cap']:9.4f} {vals['both']:9.4f} "
          f"{d['qk']:+9.4f} {d['cap']:+9.4f} {d['both']:+9.4f}")

verdict = {"seeds": list(SEEDS), "bars": {"q1": Q1_BAR, "q2": Q2_BAR,
                                          "q3_margin": Q3_MARGIN},
           "deltas": {s: d for s, d in deltas.items()}}
if len(deltas) == len(SEEDS):
    means = {k: sum(d[k] for d in deltas.values()) / len(deltas)
             for k in ("qk", "cap", "both")}
    spreads = {k: max(d[k] for d in deltas.values())
               - min(d[k] for d in deltas.values())
               for k in ("qk", "cap", "both")}
    q1 = means["both"] <= Q1_BAR
    q1b = all(d["both"] < 0 for d in deltas.values())
    q2a = means["qk"] <= Q2_BAR
    q2b = means["cap"] <= Q2_BAR
    q3 = means["both"] <= min(means["qk"], means["cap"]) + Q3_MARGIN
    print(f"  mean paired delta: qk {means['qk']:+.4f}  cap {means['cap']:+.4f}  "
          f"both {means['both']:+.4f}")
    print(f"  Q-1   {'PASS' if q1 else 'FAIL'}  bundle mean {means['both']:+.4f} "
          f"(bar {Q1_BAR:+.3f})")
    print(f"  Q-1b  {'PASS' if q1b else 'FAIL'}  both seed deltas negative")
    print(f"  Q-2a  {'PASS' if q2a else 'FAIL'}  qk mean {means['qk']:+.4f} "
          f"(bar {Q2_BAR:+.3f})")
    print(f"  Q-2b  {'PASS' if q2b else 'FAIL'}  cap mean {means['cap']:+.4f} "
          f"(bar {Q2_BAR:+.3f})")
    print(f"  Q-3   {'PASS' if q3 else 'FAIL'}  both <= better component + "
          f"{Q3_MARGIN:.3f}")
    print(f"  Q-4   spreads: qk {spreads['qk']:.4f}  cap {spreads['cap']:.4f}  "
          f"both {spreads['both']:.4f}")
    verdict.update({
        "means": means,
        "spreads": spreads,
        "q1": q1, "q1b": q1b, "q2a": q2a, "q2b": q2b, "q3": q3,
        "by_seed": by_seed,
    })
    write_json("/kaggle/working/x31_verdict.json", verdict)
    print("verdict written to /kaggle/working/x31_verdict.json")
else:
    print("no complete seed pair — verdict not computed")
    write_json("/kaggle/working/x31_verdict.json", verdict)
