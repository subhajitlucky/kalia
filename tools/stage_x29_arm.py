"""Stage one X29 arm kernel into /tmp for pushing.

Encodes the two-step stage that used to be done by hand and can silently go
wrong in exactly one way that wastes a GPU run: forgetting to patch ``ARM`` in
the runner copy. The stager patches it and verifies the result before exiting.

Usage:
    python tools/stage_x29_arm.py krama
    kaggle kernels push -p /tmp/kalia-x29-krama
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("uniform", "krama", "jata", "ghana")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS)
    args = parser.parse_args()

    dest = Path(f"/tmp/kalia-x29-{args.arm}")
    dest.mkdir(parents=True, exist_ok=True)

    runner = (ROOT / "tools" / "kalia_x29_kernel.py").read_text()
    patched, n = re.subn(r'^ARM = "\w+"', f'ARM = "{args.arm}"', runner, count=1, flags=re.M)
    assert n == 1, "ARM assignment not found in tools/kalia_x29_kernel.py"
    assert f'ARM = "{args.arm}"' in patched, "ARM patch failed"
    (dest / "kalia_x29_kernel.py").write_text(patched)

    shutil.copy2(ROOT / "kaggle" / f"kernel-metadata-x29-{args.arm}.json", dest / "kernel-metadata.json")

    meta = (dest / "kernel-metadata.json").read_text()
    assert f"kalia-x29-{args.arm}" in meta, "metadata id does not match the arm"
    print(f"staged {args.arm} -> {dest}")
    for p in sorted(dest.iterdir()):
        print(" ", p.name)


if __name__ == "__main__":
    main()
