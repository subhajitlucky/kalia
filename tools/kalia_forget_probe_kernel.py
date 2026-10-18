"""CL-0 retention probe: score one cohort of continual-update checkpoints.

Registered X29 retention endpoint -- "the CL-0 ledger
(tools/forgetting_probe.py) on each final checkpoint, drift from best-ever"
(docs/preregistrations/2026-10-11-X29-patha-replay.md). CPU kernel on purpose:
retention is a measurement, not a training run, and the 30h GPU budget stays
untouched.

A *cohort* is the explicit set of checkpoints one run scores, selected by the
``COHORT`` constant that staging patches (see ``COHORTS`` below). Today there
are two: the registered seed-1401 pāṭha ladder, and the second-seed
replication read against its seed-1401 anchors.

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
    python tools/stage_x29_forget_probe.py                          # the 1401 ladder
    python tools/stage_x29_forget_probe.py --cohort x29-replication # the replication
    kaggle kernels push -p /tmp/kalia-x29-forget-probe
"""

from __future__ import annotations

import glob
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "kalia-lm/kalia-v020"
EXPECTED_PROBE_SHA = "1664f71a4b5e96d28d20c7b531fdaf36409a25cd243dd73de0dee26bfd8fa795"
BASELINE_STEP = 4770
BASELINE_PROBE_LOSS = 2.798875
BASELINE_TOLERANCE = 0.002  # float-order wiggle on a 204,800-token mean

# Staging (tools/stage_x29_forget_probe.py) injects the repo HEAD sha here and
# refuses to stage without one, so no row can claim a revision it was not built
# from. It describes *this kernel*; the probe it runs (tools/forgetting_probe.py)
# is executed from the attached kalia-code-dev snapshot, which may lag HEAD.
# That stale-bundle class is an open 2026-10-17 review finding, not fixed here,
# and the snapshot path is printed at run start so the gap is visible.
COMMIT = "unknown"

# --- the cohort --------------------------------------------------------------
# Which checkpoints this run scores, declared rather than inferred.
#
# Until now the arm set was read off the mount-path text: the baseline was
# "every ckpt.pt without /x29- in it", and an arm's name came from
# re.search(r"x29-([a-z]+)", path). That heuristic cannot see the no-replay arm
# -- its checkpoint is out/kautilya-static/ckpt.pt -- so attaching it made the
# baseline assertion count two, and the second-seed replication could not be
# scored at all. Path heuristics are also how the wrong instrument survived
# three kernel versions (see the incident history above). The set is stated.
#
# Each arm is (label, kernel slug, checkpoint path inside that kernel's output).
COHORTS = {
    # The registered X29 ladder, seed 1401 -- the ledger read on 2026-10-13.
    "x29-1401": {
        "ledger": "forgetting_x29.jsonl",
        "arms": [
            ("uniform", "kalia-x29-uniform", "out/x29-uniform/ckpt.pt"),
            ("krama", "kalia-x29-krama", "out/x29-krama/ckpt.pt"),
            ("jata", "kalia-x29-jata", "out/x29-jata/ckpt.pt"),
            ("ghana", "kalia-x29-ghana", "out/x29-ghana/ckpt.pt"),
        ],
    },
    # The second-seed replication and the seed-1401 anchors it is read against:
    # replay (uniform, 10%) vs no-replay (static) at both seeds -- the 2x2 that
    # decides whether the X29 direction reproduces. The X28 static arm has never
    # been scored on CL-0 at either seed, so without the static-s1401 anchor the
    # no-replay side of the contrast would have no seed-1401 row to pair with.
    # uniform-s1401 is the within-session anchor: the 2026-10-13 ledger recorded
    # it at 2.775169, and a different number here means the instrument moved,
    # not the model.
    #
    # Caveat, to be carried with the result: replay vs no-replay is not the only
    # difference between these arms. uniform samples the concatenated train.bin
    # and adds a replay shard; static samples the four per-source shards
    # directly at the corpus's own 60/20/15/5 weights with adaptive_every: 0.
    # Same distribution, same hyperparameters, two delivery paths -- so this
    # contrast is replay-plus-path, not replay alone.
    "x29-replication": {
        "ledger": "forgetting_x29_replication.jsonl",
        "arms": [
            ("uniform-s1401", "kalia-x29-uniform", "out/x29-uniform/ckpt.pt"),
            ("static-s1401", "kalia-x28-static", "out/kautilya-static/ckpt.pt"),
            ("uniform-s1402", "kalia-x29-uniform-s1402", "out/x29-uniform/ckpt.pt"),
            ("static-s1402", "kalia-x28-static-s1402", "out/kautilya-static/ckpt.pt"),
        ],
    },
}

COHORT = "x29-1401"  # staging patches this
KERNEL_MOUNT = "/kaggle/input/notebooks"  # where Kaggle mounts kernel_sources
BASELINE_GLOB = "/kaggle/input/datasets/*/kalia-v020-ckpt/ckpt.pt"
ATTACHED_GLOB = "/kaggle/input/**/ckpt.pt"  # every checkpoint Kaggle mounted


def resolve_cohort(
    name: str,
    mount: str = KERNEL_MOUNT,
    baseline_glob: str = BASELINE_GLOB,
    attached_glob: str = ATTACHED_GLOB,
) -> tuple[Path, list[tuple[str, Path]]]:
    """Return (baseline checkpoint, [(label, checkpoint), ...]) or fail loudly.

    Three ways this refuses to score: a baseline that is not exactly one file, a
    declared arm whose checkpoint is missing or ambiguous, and an attached
    checkpoint that no arm declares. The last is the guard against a stale
    kernel_sources entry quietly joining a ledger under a name nobody chose.
    """
    spec = COHORTS[name]
    baselines = sorted(glob.glob(baseline_glob))
    assert len(baselines) == 1, (
        f"expected exactly one v0.2.0 baseline at {baseline_glob}, got {baselines}"
    )
    arms: list[tuple[str, Path]] = []
    for label, slug, rel in spec["arms"]:
        hits = sorted(glob.glob(f"{mount}/*/{slug}/{rel}"))
        assert len(hits) == 1, (
            f"arm {label}: expected exactly one {slug}/{rel} under {mount}, "
            f"got {hits} -- is {slug} attached as a kernel source?"
        )
        arms.append((label, Path(hits[0])))
    declared = {str(baselines[0])} | {str(p) for _, p in arms}
    undeclared = sorted(set(glob.glob(attached_glob, recursive=True)) - declared)
    assert not undeclared, (
        f"checkpoints are attached that cohort {name!r} does not declare: {undeclared} "
        "-- remove the stale kernel source rather than letting it into the ledger"
    )
    return Path(baselines[0]), arms

# --- the code, same resolution as every other kernel ------------------------
# The attached arm kernels each carry a full copy of the code tree in their
# output (the runner chdir's into /kaggle/working/kalia), so this glob matches
# several patha.py and taking the first was only ever correct because
# "datasets" sorts before "notebooks". Prefer the real snapshot explicitly.
hits = sorted(glob.glob("/kaggle/input/**/patha.py", recursive=True))
preferred = [p for p in hits if "/kalia-code-dev/" in p]
assert preferred, (
    "kalia-code-dev not attached or lacks patha.py; other matches "
    "(do NOT trust them, they are stale bundles): " + str(hits)
)
code_root = Path(preferred[0]).parent
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


# --- resolve this cohort's checkpoints ---------------------------------------
assert COHORT in COHORTS, (
    f"unknown cohort {COHORT!r}; staging patches COHORT, choices are {sorted(COHORTS)}"
)
LEDGER_NAME = COHORTS[COHORT]["ledger"]
print(f"cohort: {COHORT} -> {LEDGER_NAME}")
baseline_ckpt, arm_ckpts = resolve_cohort(COHORT)
print(f"baseline: {baseline_ckpt}")
for label, ckpt in arm_ckpts:
    print(f"  arm {label}: {ckpt}")

# --- self-check: the baseline must reproduce its recorded probe loss --------
selfcheck = probe_dir / "selfcheck.jsonl"
selfcheck.write_text(ledger.read_text())
check_row = run_probe(baseline_ckpt, selfcheck)
delta = abs(check_row["probe_loss"] - BASELINE_PROBE_LOSS)
print(f"\nself-check: v0.2.0 re-scored {check_row['probe_loss']:.6f} "
      f"vs recorded {BASELINE_PROBE_LOSS:.6f} (delta {delta:.6f})")
assert delta <= BASELINE_TOLERANCE, (
    "baseline does not reproduce; the probe, checkpoint, or environment is not "
    "the frozen instrument"
)

# --- score the arms ----------------------------------------------------------
for label, ckpt in arm_ckpts:
    row = run_probe(ckpt, ledger)
    print(f"arm {label}: probe_loss {row['probe_loss']:.6f}  drift {row['drift_from_best']:+.6f}")

# --- archive the ledger ------------------------------------------------------
out = Path("/kaggle/working") / LEDGER_NAME
shutil.copy(ledger, out)
print(f"\n=== CL-0 ledger ({COHORT}) ===")
print("drift is sequential (against the best row so far), so only probe_loss "
      "is comparable across arms; re-reference against the baseline row.")
for line in ledger.read_text().strip().splitlines():
    r = json.loads(line)
    drift = f"{r['drift_from_best']:+.6f}" if r["drift_from_best"] is not None else "   baseline"
    print(f"  step {r['step']:>5}  probe_loss {r['probe_loss']:.6f}  drift {drift}")
print(f"archived: {out}")
