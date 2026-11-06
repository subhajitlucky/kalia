"""Kaggle CPU kernel: generate KALIA-Vision v1 two-object data."""
from __future__ import annotations
import json, os, shutil, subprocess, sys
from pathlib import Path

hits = sorted(Path("/kaggle/input").rglob("synthetic_vision_v1.py"))
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow"], check=True)
out = Path("/kaggle/working/vision-v1-data")
subprocess.run([sys.executable, "synthetic_vision_v1.py", "--out-dir", str(out),
                "--n-train", "50000", "--n-val", "5000", "--n-test", "5000"], check=True)
print((out / "audit.json").read_text()[:1000])
