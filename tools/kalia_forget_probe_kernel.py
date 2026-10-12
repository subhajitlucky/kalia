"""X29 retention probe: score every attached arm checkpoint on the frozen CL-0 probe.

Registered X29 retention endpoint -- "the CL-0 ledger
(tools/forgetting_probe.py) on each final checkpoint, drift from best-ever"
(docs/preregistrations/2026-10-11-X29-patha-replay.md). CPU kernel on purpose:
retention is a measurement, not a training run, and the 30h GPU budget stays
untouched.

History worth keeping (v1-v3 of this kernel scored the WRONG instrument):

  The CL-0 ledger's baseline row (step 4770, probe_loss 2.798875) was measured
  on 2026-10-07 on ``corpus/probe_val.bin`` -- the G2-pinned probe shard from
  the corpus publish, sha256 1664f71a..., pinned in
  docs/legal/val-shard-hashes.json. The earlier versions of this kernel scored
  ``data/probe/val_forget.bin`` (sha 7b71e339...), a different split made on
  2026-09-28 for the pretrain-era forgetting curve. Re-scoring v0.2.0 on the
  wrong file gave 2.909614 (+0.1107 vs the recorded row), and the self-check
  refused to score any arm -- correctly. The same producer/consumer probe-file
  mismatch has a recorded precedent in test_publish_contract.py ("producer
  wrote val_forget.bin, consumer wanted probe_val.bin"); this kernel is the
  third sighting, and the fix is the same: the pinned shard, fetched from the
  model repo, verified against the G2 record at every run.

Guards, in order of paranoia:
  1. the probe shard is fetched from kalia-lm/kalia-v020 and refused unless its
     sha256 matches the G2 record in docs/legal/val-shard-hashes.json;
  2. the ledger is seeded with the exact v0.2.0 baseline row (2.798875), so
     "best ever" is the registered number;
  3. the v0.2.0 checkpoint is re-scored first (separate self-check ledger) and
     must reproduce the recorded baseline; a failure means the probe, the
     checkpoint, or the environment is not the instrument, and nothing else is
     believable.

Usage (from the repo root):
    python tools/stage_x29_forget_probe.py
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

REPO = "kalia-lm/kalia-v020"
EXPECTED_PROBE_SHA = "1664f71a4b5e96d28d20c7b531fdaf36409a25cd243dd73de0dee26bfd8fa795"
BASELINE_STEP = 4770
BASELINE_PROBE_LOSS = 2.798875
BASELINE_TOLERANCE = 0.002  # float-order wiggle on a 204,800-token mean
COMMIT = "d220918"

# --- the code, same resolution as every other kernel ------------------------
hits = sorted(glob.glob("/kaggle/input/**/patha.py", recursive=True))
assert hits, "attach subhajitlucky/kalia-code-dev"
code_root = Path(hits[0]).parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
for req in ("train.py", "tools/forgetting_probe.py", "configs/kalia-v020.yaml", "patha.py"):
    assert (work / req).exists(), f"{req} missing from the copied code snapshot"
print(f"code: {code_root}")

# --- the frozen probe shard, fetched and G2-verified ------------------------
probe_hits = sorted(glob.glob("/kaggle/input/**/probe_val.bin", recursive=True))
probe_bin = probe_hits[0] if probe_hits else None
if probe_bin:
    print(f"probe shard: attached dataset file {probe_bin}")
if probe_bin is None:
    from huggingface_hub import hf_hub_download

    probe_bin = hf_hub_download(REPO, "corpus/probe_val.bin", repo_type="model")
    print(f"probe shard: hub pull {REPO}/corpus/probe_val.bin")
digest = hashlib.sha256(open(probe_bin, "rb").read()).hexdigest()
assert digest == EXPECTED_PROBE_SHA, (
    f"probe shard does not match the G2-recorded yardstick ({digest}); stop -- "
    "the CL-0 ledger would be comparing different data across updates"
)
print(f"probe sha256 ok ({digest[:12]}...)")

# --- seed the CL-0 ledger with the exact registered baseline row ------------
probe_dir = Path("/kaggle/working/probe")
probe_dir.mkdir(exist_ok=True)
ledger = probe_dir / "cl0_ledger.jsonl"
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


def run_probe(ckpt: Path, target_ledger: Path) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "tools/forgetting_probe.py",
            "--ckpt", str(ckpt),
            "--probe", str(probe_bin),
            "--ledger", str(target_ledger),
            "--batches", "25",
            "--batch-size", "8",
            "--device", "cpu",
            "--commit", COMMIT,
        ],
        cwd=work,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        print(result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
        result.check_returncode()
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
check_row = run_probe(Path(baseline_ckpts[0]), selfcheck)
delta = abs(check_row["probe_loss"] - BASELINE_PROBE_LOSS)
print(f"\nself-check: v0.2.0 re-scored {check_row['probe_loss']:.6f} "
      f"vs recorded {BASELINE_PROBE_LOSS:.6f} (delta {delta:.6f})")
assert delta <= BASELINE_TOLERANCE, (
    "baseline does not reproduce; the probe, checkpoint, or environment is not "
    "the frozen instrument"
)

# --- score the arms ----------------------------------------------------------
for ckpt in arm_ckpts:
    arm = re.search(r"x29-([a-z]+)", ckpt).group(1)
    row = run_probe(Path(ckpt), ledger)
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
