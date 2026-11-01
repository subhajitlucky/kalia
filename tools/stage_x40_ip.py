"""Stage the X40 30M kernel into /tmp for pushing.

One kernel runs all five arms and both seeds, so there is no per-arm constant
to pin beyond the identity. The two constants this script does pin are the ones
the run manifest must report: ``KERNEL_ID`` (the Kaggle kernel it actually runs
as) and ``KERNEL_REVISION`` (the repo HEAD that produced the runner).

``KERNEL_REVISION`` is the repo HEAD; a dirty runner stages as ``<sha>+dirty``
rather than being refused, because an uncommitted experiment should still be
runnable as long as the manifest does not claim a clean revision. The verified
path is to commit first.

Usage:
    python tools/stage_x40_ip.py
    kaggle kernels push -p /tmp/kalia-x40-ip
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = "kalia_x40_ip_kernel.py"
META = "kernel-metadata-x40-ip.json"


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def pin(text: str, name: str, value: str) -> str:
    patched, n = re.subn(rf'^{name} = ".*?"', f'{name} = "{value}"', text, count=1, flags=re.M)
    assert n == 1, f"{name} assignment not found (or found {n} times)"
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("/tmp/kalia-x40-ip"))
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    meta_src = ROOT / "kaggle" / META
    if not meta_src.exists():
        raise SystemExit(f"missing committed metadata kaggle/{META}")
    meta = json.loads(meta_src.read_text())
    if meta["code_file"] != RUNNER:
        raise SystemExit(f"{META}: code_file {meta['code_file']} != {RUNNER}")
    if meta["id"] != "subhajitlucky/kalia-x40-ip":
        raise SystemExit(f"{META}: unexpected id {meta['id']}")

    rel = f"tools/{RUNNER}"
    dirty = git("status", "--porcelain", "--", rel)
    if dirty and not args.allow_dirty:
        raise SystemExit(f"{rel} has uncommitted changes:\n{dirty}\ncommit them first")
    revision = git("rev-parse", "HEAD") + ("+dirty" if dirty else "")

    patched = (ROOT / rel).read_text()
    patched = pin(patched, "KERNEL_ID", meta["id"])
    patched = pin(patched, "KERNEL_REVISION", revision)
    expected = {"KERNEL_ID": meta["id"], "KERNEL_REVISION": revision}

    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / RUNNER).write_text(patched)
    shutil.copy2(meta_src, args.dest / "kernel-metadata.json")

    staged = (args.dest / RUNNER).read_text()
    for name, value in expected.items():
        if f'{name} = "{value}"' not in staged:
            raise SystemExit(f"staged {name} is not {value!r} -- refusing a silent mis-stage")

    print(f"staged X40 30M @ {revision[:12]} -> {args.dest}")
    for name, value in expected.items():
        print(f"  {name} = {value}")
    print(f"  metadata: {META}")
    print(f"\n  kaggle kernels push -p {args.dest}")


if __name__ == "__main__":
    main()
