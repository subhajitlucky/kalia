"""Stage one arm kernel into /tmp for pushing.

Encodes the two-step stage that used to be done by hand and can silently go
wrong in exactly one way that wastes a GPU run: forgetting to patch a constant
in the runner copy. The stager pins five of them — ``ARM``, ``SEED``, ``REPLAY``
where the runner has one, ``KERNEL_ID`` and ``KERNEL_REVISION`` — verifies each
patch landed, and asserts the committed metadata agrees with the arm and seed
before exiting.

``KERNEL_REVISION`` is the repo HEAD, and a dirty runner stages as ``<sha>+dirty``
rather than being refused: the manifest the run writes must not claim a revision
that never ran, but neither should an uncommitted experiment be unstoppable.
``--allow-dirty`` is the explicit form of the same thing when the guard is on.

Usage:
    python tools/stage_x29_arm.py krama
    python tools/stage_x29_arm.py uniform --seed 1402
    python tools/stage_x29_arm.py static --kernel x28 --seed 1402
    python tools/stage_x29_arm.py noreplay --seed 1401   # the 2026-10-18 control
    kaggle kernels push -p /tmp/kalia-x29-noreplay
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

KERNELS = {
    "x29": {
        "runner": "kalia_x29_kernel.py",
        # arm -> the replay fraction staged into REPLAY. `noreplay` is the
        # 2026-10-18 single-variable control: same runner, corpus and step
        # budget as `uniform`, replay at zero, so replay is the only move.
        "arms": {
            "uniform": "0.10",
            "krama": "0.10",
            "jata": "0.10",
            "ghana": "0.10",
            "noreplay": "0.00",
        },
        # Constants this runner declares, beyond ARM/KERNEL_ID/KERNEL_REVISION
        # which every runner has. Declared rather than assumed: pinning a name a
        # runner does not define fails, and inventing a knob it lacks would be a
        # lie about what the run can vary.
        "pins": ("SEED", "REPLAY"),
    },
    # The Kautilya trio has no replay knob: it samples the four published
    # per-source shards and passes no --replay-bin, so replay_prob stays at its
    # 0.0 default. None means "this runner has no REPLAY constant to pin".
    "x28": {
        "runner": "kalia_kautilya_trio_kernel.py",
        "arms": {"static": None, "level": None, "trend": None},
        "pins": ("SEED",),
    },
    # X17 runs its whole registered experiment in one kernel -- four arms, two
    # seeds -- so there is no per-arm SEED to stage. Its seed pair is a kernel
    # constant under the amendment (1338/1339).
    "x17": {
        "runner": "kalia_x17_kernel.py",
        "arms": {"docmask": None},
        "pins": (),
    },
    # The 2x2 re-evaluation: scores the four archived X17 checkpoints under both
    # evaluation settings, on CPU. Same shape as x17 -- one kernel, no per-arm
    # seed -- so nothing to pin but the identity.
    "x17eval": {
        "runner": "kalia_x17_eval_kernel.py",
        "arms": {"mask2x2": None},
        "pins": (),
    },
    # Entity decoding, tested on a held-out half of each continuation. Same
    # shape again: one kernel, no per-arm seed.
    "entity": {
        "runner": "kalia_entity_holdout_kernel.py",
        "arms": {"holdout": None},
        "pins": (),
    },
    # X31 runs the whole 2x2 at two seeds in one kernel. There is no per-arm
    # seed to stage; the seed pair is a kernel constant under the registration.
    "x31": {
        "runner": "kalia_x31_kernel.py",
        "arms": {"qk-isolation": None},
        "pins": (),
    },
    # The 2,000-sample benchmark pair. CPU, so the arm name is just the
    # resolution ("2k"), not a seed.
    "bench-pair": {
        "runner": "kalia_bench_pair_kernel.py",
        "arms": {"2k": None},
        "pins": (),
    },
    # The mechanistic probe: one kernel, one arm, no per-arm seed.
    "x32": {
        "runner": "kalia_mech_kernel.py",
        "arms": {"entity-circuit": None},
        "pins": (),
    },
    # The linear-readout probe: one kernel, one arm, no per-arm seed.
    "x33": {
        "runner": "kalia_x33_kernel.py",
        "arms": {"linear-readout": None},
        "pins": (),
    },
    # X34 replicates X33 with a fresh generation seed. The kernel constant is a
    # string so the stager can pin it without touching the probe.
    "x34": {
        "runner": "kalia_x33_kernel.py",
        "arms": {"seed": None},
        "pins": ("SEED_BASE",),
    },
}


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def pin(text: str, name: str, value: str) -> str:
    """Pin one module-level string constant, refusing anything ambiguous."""
    patched, n = re.subn(rf'^{name} = ".*?"', f'{name} = "{value}"', text, count=1, flags=re.M)
    assert n == 1, f"{name} assignment not found (or found {n} times)"
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm")
    parser.add_argument("--kernel", choices=sorted(KERNELS), default="x29")
    parser.add_argument("--seed", default="1401", help="1401 is the original; 1402 is the registered replication")
    parser.add_argument("--seed-base", default="7",
                        help="value for SEED_BASE when the kernel spec pins it (X34 uses 1007)")
    parser.add_argument("--dest", type=Path, default=None, help="override the /tmp staging directory")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="stage despite an uncommitted runner (revision is stamped +dirty)")
    args = parser.parse_args()

    spec = KERNELS[args.kernel]
    if args.arm not in spec["arms"]:
        raise SystemExit(f"{args.kernel} arms are {sorted(spec['arms'])}, not {args.arm!r}")
    suffix = "" if args.seed == "1401" else f"-s{args.seed}"
    dest = args.dest or Path(f"/tmp/kalia-{args.kernel}-{args.arm}{suffix}")

    # The metadata is the second place an arm's identity lives; it is read
    # first so the runner can be stamped with the id it will actually run as.
    meta_name = f"kernel-metadata-{args.kernel}-{args.arm}{suffix}.json"
    meta_src = ROOT / "kaggle" / meta_name
    if not meta_src.exists():
        raise SystemExit(f"missing committed metadata kaggle/{meta_name}")
    meta = json.loads(meta_src.read_text())
    if meta["code_file"] != spec["runner"]:
        raise SystemExit(f"{meta_name}: code_file {meta['code_file']} != {spec['runner']}")
    expected_id = f"subhajitlucky/kalia-{args.kernel}-{args.arm}{suffix}"
    if meta["id"] != expected_id:
        raise SystemExit(f"{meta_name}: id {meta['id']} != {expected_id}")

    rel = f"tools/{spec['runner']}"
    dirty = git("status", "--porcelain", "--", rel)
    if dirty and not args.allow_dirty:
        raise SystemExit(f"{rel} has uncommitted changes:\n{dirty}\ncommit them first")
    revision = git("rev-parse", "HEAD") + ("+dirty" if dirty else "")

    patched = (ROOT / rel).read_text()
    patched = pin(patched, "ARM", args.arm)
    patched = pin(patched, "KERNEL_ID", meta["id"])
    patched = pin(patched, "KERNEL_REVISION", revision)
    expected = {"ARM": args.arm, "KERNEL_ID": meta["id"], "KERNEL_REVISION": revision}
    if "SEED" in spec["pins"]:
        patched = pin(patched, "SEED", args.seed)
        expected["SEED"] = args.seed
    if "SEED_BASE" in spec["pins"]:
        patched = pin(patched, "SEED_BASE", args.seed_base)
        expected["SEED_BASE"] = args.seed_base
    replay = spec["arms"][args.arm]
    if "REPLAY" in spec["pins"] and replay is not None:
        patched = pin(patched, "REPLAY", replay)
        expected["REPLAY"] = replay

    dest.mkdir(parents=True, exist_ok=True)
    (dest / spec["runner"]).write_text(patched)
    shutil.copy2(meta_src, dest / "kernel-metadata.json")

    # Read the staged file back: a pin that silently did not land is the one
    # failure this script exists to make impossible.
    staged = (dest / spec["runner"]).read_text()
    for name, value in expected.items():
        if f'{name} = "{value}"' not in staged:
            raise SystemExit(f"staged {name} is not {value!r} -- refusing to leave a silent mis-stage")

    print(f"staged {args.kernel} {args.arm} seed {args.seed} @ {revision[:12]} -> {dest}")
    for name, value in expected.items():
        print(f"  {name} = {value}")
    print(f"  metadata: {meta_name}")
    print(f"\n  kaggle kernels push -p {dest}")


if __name__ == "__main__":
    main()
