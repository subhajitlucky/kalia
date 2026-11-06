"""Kaggle CPU kernel: generate the KALIA-Vision v0 synthetic dataset."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

hits = sorted(Path("/kaggle/input").rglob("synthetic_vision.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))
try:
    import PIL  # noqa: F401
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow"],
                   check=True)
out = Path("/kaggle/working/vision-data")
subprocess.run([sys.executable, "synthetic_vision.py", "--out-dir", str(out),
                "--n-train", "50000", "--n-val", "5000", "--n-test", "5000"],
               check=True)
audit = json.loads((out / "audit.json").read_text())
print(json.dumps({k: v["n"] for k, v in audit["splits"].items()}, indent=2))
