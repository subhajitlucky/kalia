"""X29 retention probe: score every attached arm checkpoint on the frozen CL-0 probe.

This is the registered X29 retention endpoint -- "the CL-0 ledger
(tools/forgetting_probe.py) on each final checkpoint, drift from best-ever"
(docs/preregistrations/2026-10-11-X29-patha-replay.md). It runs as a CPU
kernel on purpose: retention is a measurement, not a training run, and the
30h weekly GPU budget stays untouched. The X29 arms' checkpoints live in
their kernel outputs; kernel_sources mounts them here.

Correctness guards, in order of paranoia:
  1. the frozen probe file is rebuilt from the canonical val.bin with the
     deterministic document-boundary splitter, then refused unless its
     sha256 matches the instrument frozen on 2026-09-28;
  2. the ledger is seeded with the exact v0.2.0 baseline row (step 4770,
     probe_loss 2.798875) so "best ever" means the registered number;
  3. the v0.2.0 checkpoint is re-scored first and the fresh value must
     reproduce the recorded baseline; a failure means the probe, the data,
     or the loader is not the instrument, and nothing else is believable.

Every km eval in this project is cheap on CPU and expensive on GPU attention;
this kernel is the pattern for keeping it that way.

Usage (from the repo root):
    python tools/stage_x29_forget_probe.py     # stages /tmp/kalia-x29-forget-probe
    kaggle kernels push -p /tmp/kalia-x29-forget-probe
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_PROBE_SHA = "7b71e339fab41a5ca54ed013c08ac7881dd4bdcd4fc3da92097ec3b6b1aa3187"
BASELINE_STEP = 4770
BASELINE_PROBE_LOSS = 2.798875
BASELINE_TOLERANCE = 0.002  # cross-device float wiggle on a 204,800-token mean
COMMIT = "d1055c5"

# --- the code, same resolution as every other kernel ------------------------
hits = sorted(glob.glob("/kaggle/input/**/patha.py", recursive=True))
assert hits, "attach subhajitlucky/kalia-code-dev"
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(hits[0].parent, work)
print(f"code: {hits[0].parent}")

# --- rebuild the frozen probe and verify it is the instrument ---------------
val_bins = sorted(glob.glob("/kaggle/input/**/val.bin", recursive=True))
assert len(val_bins) == 1, f"expected exactly one val.bin, got {val_bins}"
val_bin = val_bins[0]
probe_dir = Path("/kaggle/working/probe")
if not probe_dir.exists():
    subprocess.run(
        [sys.executable, "tools/build_forget_probe.py", "--val-bin", val_bin, "--out-dir", str(probe_dir)],
        cwd=work,
        check=True,
    )
digest = hashlib.sha256((probe_dir / "val_forget.bin").read_bytes()).hexdigest()
assert digest == EXPECTED_PROBE_SHA, f"rebuilt probe differs from the frozen instrument: {digest}"
print(f"val_forget.bin sha256 ok ({digest[:12]}...), rebuilt from {val_bin}")

# --- seed the CL-0 ledger with the exact registered baseline ----------------
ledger = probe_dir / "forgetting.jsonl"
baseline_row = {
    "step": BASELINE_STEP,
    "probe_loss": BASELINE_PROBE_LOSS,
    "probe_tokens": 204800,
    "batches": 25,
    "seed": 1234,
    "drift_from_best": None,
    "bpB": 0.933016,
    "bytes_per_token": 4.32782,
    "commit": "v0.2.0 baseline",
    "recorded": "2026-10-07T16:18:06+00:00",
}
ledger.write_text(json.dumps(baseline_row) + "\n")


def run_probe(ckpt: Path, step: int, target_ledger: Path) -> dict:
    subprocess.run(
        [
            sys.executable,
            "tools/forgetting_probe.py",
            "--ckpt", str(ckpt),
            "--step", str(step),
            "--probe", str(probe_dir / "val_forget.bin"),
            "--ledger", str(target_ledger),
            "--device", "cpu",
            "--commit", COMMIT,
        ],
        cwd=work,
        check=True,
    )
    return json.loads(target_ledger.read_text().strip().splitlines()[-1])


# --- locate checkpoints ------------------------------------------------------
all_ckpts = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
baseline_ckpts = [p for p in all_ckpts if "/x29-" not in p]
arm_ckpts = [p for p in all_ckpts if "/x29-" in p]
assert len(baseline_ckpts) == 1, f"expected one baseline ckpt, got {baseline_ckpts}"
assert arm_ckpts, "no x29 arm checkpoints attached"
print(f"baseline: {baseline_ckpts[0]}")
for p in arm_ckpts:
    print(f"  arm: {p}")

# --- self-check: the baseline must reproduce its recorded probe loss --------
selfcheck = probe_dir / "selfcheck.jsonl"
selfcheck.write_text(ledger.read_text())
check_row = run_probe(Path(baseline_ckpts[0]), BASELINE_STEP, selfcheck)
delta = abs(check_row["probe_loss"] - BASELINE_PROBE_LOSS)
print(f"\nself-check: v0.2.0 re-scored {check_row['probe_loss']:.6f} "
      f"vs recorded {BASELINE_PROBE_LOSS:.6f} (delta {delta:.6f})")
assert delta <= BASELINE_TOLERANCE, (
    "baseline does not reproduce; the probe/data/loader is not the frozen instrument"
)

# --- score the arms ----------------------------------------------------------
for ckpt in arm_ckpts:
    arm = re.search(r"x29-([a-z]+)", ckpt).group(1)
    row = run_probe(Path(ckpt), 5770, ledger)
    print(f"arm {arm}: probe_loss {row['probe_loss']:.6f}  drift {row['drift_from_best']:+.6f}")

# --- archive the ledger ------------------------------------------------------
out = Path("/kaggle/working/forgetting_x29.jsonl")
shutil.copy(ledger, out)
print("\n=== CL-0 ledger (X29) ===")
for line in ledger.read_text().strip().splitlines():
    r = json.loads(line)
    drift = f"{r['drift_from_best']:+.6f}" if r["drift_from_best"] is not None else "   baseline"
    print(f"  step {r['step']:>5}  probe_loss {r['probe_loss']:.6f}  drift {drift}")
print(f"archived: {out}")
