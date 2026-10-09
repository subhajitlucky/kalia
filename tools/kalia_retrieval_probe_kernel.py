"""Kaggle CPU kernel: run the retrieval-utilization probe against v0.2.0.

Script kernel, no GPU, no secrets. Expects three inputs:
- ``kalia-code-dev`` dataset (this code; preferred over any stale bundle),
- ``kalia-v020-ckpt`` dataset (the checkpoint, so no 334MB hub pull),
- ``kalia-prep-v2b`` kernel source (``val.bin`` for the distraction arm).

The harness itself is ``retrieval_probe.py``; this file only locates inputs,
asserts the exact files it invokes, and runs it.
"""

import glob
import os
import shutil
import subprocess
import sys

work = "/kaggle/working/kalia"
if not os.path.exists(work):
    hits = sorted(glob.glob("/kaggle/input/**/retrieval_probe.py", recursive=True))
    preferred = [p for p in hits if "/kalia-code-dev/" in p]
    assert preferred, (
        "kalia-code-dev not attached or lacks retrieval_probe.py; other matches "
        "(do NOT trust them, they are stale bundles): " + str(hits)
    )
    shutil.copytree(os.path.dirname(preferred[0]), work)
os.chdir(work)
for req in ("retrieval_probe.py", "retrieval.py", "best_of_n.py", "model.py", "sample.py"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"

try:
    import tiktoken  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"], check=True)

ckpt_hits = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
assert ckpt_hits, "attach subhajitlucky/kalia-v020-ckpt"
val_hits = sorted(glob.glob("/kaggle/input/**/val.bin", recursive=True))
assert val_hits, "attach kalia-prep-v2b as a kernel source for val.bin"
print("checkpoint:", ckpt_hits[0])
print("val shard:", val_hits[0])

cmd = [
    sys.executable,
    "retrieval_probe.py",
    "--ckpt", ckpt_hits[0],
    "--val-bin", val_hits[0],
    "--out", "/kaggle/working/retrieval_probe.json",
    "--facts", "64",
    "--distract", "48",
    "--n", "8",
    "--device", "cpu",
]
r = subprocess.run(cmd, text=True)
sys.exit(r.returncode)
