"""Stage one arm or second-seed replica kernel into /tmp for pushing.

Encodes the two-step stage that used to be done by hand and can silently go
wrong in exactly one way that wastes a GPU run: forgetting to patch ``ARM``
(or ``SEED``) in the runner copy. The stager patches both and verifies the
result, the committed metadata and the kernel id agree before exiting.

Usage:
    python tools/stage_x29_arm.py krama
    python tools/stage_x29_arm.py uniform --seed 1402
    python tools/stage_x29_arm.py static --kernel x28 --seed 1402
    kaggle kernels push -p /tmp/kalia-x29-uniform-s1402
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

KERNELS = {
    "x29": {
        "runner": "kalia_x29_kernel.py",
        "arms": ("uniform", "krama", "jata", "ghana"),
    },
    "x28": {
        "runner": "kalia_kautilya_trio_kernel.py",
        "arms": ("static", "level", "trend"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm")
    parser.add_argument("--kernel", choices=sorted(KERNELS), default="x29")
    parser.add_argument("--seed", default="1401", help="1401 is the original; 1402 is the registered replication")
    parser.add_argument("--dest", type=Path, default=None, help="override the /tmp staging directory")
    args = parser.parse_args()

    spec = KERNELS[args.kernel]
    assert args.arm in spec["arms"], f"{args.kernel} arms are {spec['arms']}, not {args.arm}"
    suffix = "" if args.seed == "1401" else f"-s{args.seed}"
    dest = args.dest or Path(f"/tmp/kalia-{args.kernel}-{args.arm}{suffix}")
    dest.mkdir(parents=True, exist_ok=True)

    runner = (ROOT / "tools" / spec["runner"]).read_text()
    patched, n = re.subn(r'^ARM = "\w+"', f'ARM = "{args.arm}"', runner, count=1, flags=re.M)
    assert n == 1, f"ARM assignment not found in tools/{spec['runner']}"
    patched, m = re.subn(r'^SEED = "\w+"', f'SEED = "{args.seed}"', patched, count=1, flags=re.M)
    assert m == 1, f"SEED assignment not found in tools/{spec['runner']}"
    (dest / spec["runner"]).write_text(patched)

    meta_name = f"kernel-metadata-{args.kernel}-{args.arm}{suffix}.json"
    meta_src = ROOT / "kaggle" / meta_name
    assert meta_src.exists(), f"missing committed metadata kaggle/{meta_name}"
    meta = json.loads(meta_src.read_text())
    assert meta["code_file"] == spec["runner"], f"{meta_name}: code_file {meta['code_file']} != {spec['runner']}"
    expected_id = f"subhajitlucky/kalia-{args.kernel}-{args.arm}{suffix}"
    assert meta["id"] == expected_id, f"{meta_name}: id {meta['id']} != {expected_id}"
    shutil.copy2(meta_src, dest / "kernel-metadata.json")

    print(f"staged {args.kernel} {args.arm} seed {args.seed} -> {dest}")
    for p in sorted(dest.iterdir()):
        print(" ", p.name)


if __name__ == "__main__":
    main()
