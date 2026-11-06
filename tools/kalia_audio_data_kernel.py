"""Kaggle CPU kernel: generate synthetic spoken commands."""
from __future__ import annotations
import json, os, shutil, subprocess, sys
from pathlib import Path

hits = sorted(Path("/kaggle/input").rglob("synthetic_audio.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))
out = Path("/kaggle/working/audio-data")
subprocess.run([sys.executable, "synthetic_audio.py", "--out-dir", str(out),
                "--n-train", "5000", "--n-val", "1000", "--n-test", "1000"],
               check=True)
print((out / "audit.json").read_text()[:1000])
