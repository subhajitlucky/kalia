"""Stage the X29 forget-probe kernel into /tmp for pushing.

The probe is re-pushed every time a new X29 arm finishes (kernel_sources fix
which checkpoints exist), so staging is a script rather than a remembered
sequence of copies.

Usage:
    python tools/stage_x29_forget_probe.py
    kaggle kernels push -p /tmp/kalia-x29-forget-probe
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = Path("/tmp/kalia-x29-forget-probe")

DEST.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / "tools" / "kalia_forget_probe_kernel.py", DEST / "kalia_forget_probe_kernel.py")
shutil.copy2(ROOT / "kaggle" / "kernel-metadata-x29-forget-probe.json", DEST / "kernel-metadata.json")
print(f"staged -> {DEST}")
for p in sorted(DEST.iterdir()):
    print(" ", p.name)
