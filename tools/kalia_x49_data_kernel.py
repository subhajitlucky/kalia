"""Kaggle CPU kernel: X49 matched data mixtures.

Builds two 524M-token training sets from the same source shards:
- control: 60 FineWeb-Edu / 20 TinyStories / 15 Cosmopedia / 5 code
- treatment: 45 FineWeb-Edu / 5 TinyStories / 40 Cosmopedia / 10 code

The control validation shard is shared by both arms. CPU only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TRAIN_TOKENS = 524_000_000
VAL_TOKENS = 10_000_000

hits = sorted(Path("/kaggle/input").rglob("mix_bins.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))
from mix_bins import mix_shards  # noqa: E402

sources = {}
for name in ("tinystories.bin", "smollm_fineweb_edu.bin", "cosmopedia.bin", "stack_smol.bin"):
    found = sorted(Path("/kaggle/input").rglob(name))
    assert found, f"{name} not attached; attach kalia-prep-v2"
    sources[name] = found[0]
print("sources:", {k: str(v) for k, v in sources.items()}, flush=True)

out = Path("/kaggle/working/x49-data")
out.mkdir(parents=True, exist_ok=True)

configs = {
    "control": [
        (sources["tinystories.bin"], 20.0),
        (sources["smollm_fineweb_edu.bin"], 60.0),
        (sources["cosmopedia.bin"], 15.0),
        (sources["stack_smol.bin"], 5.0),
    ],
    "treatment": [
        (sources["tinystories.bin"], 5.0),
        (sources["smollm_fineweb_edu.bin"], 45.0),
        (sources["cosmopedia.bin"], 40.0),
        (sources["stack_smol.bin"], 10.0),
    ],
}

manifest = {
    "experiment": "X49-data",
    "train_tokens": TRAIN_TOKENS,
    "val_tokens": VAL_TOKENS,
    "sources": {k: str(v) for k, v in sources.items()},
    "mixtures": {},
}
for name, shards in configs.items():
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===", flush=True)
    stats = mix_shards(
        shards, d / "val.bin", d / "train.bin",
        VAL_TOKENS, TRAIN_TOKENS,
    )
    manifest["mixtures"][name] = stats
    for f in ("train.bin", "val.bin"):
        p = d / f
        manifest["mixtures"][name][f] = {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    print(json.dumps(stats, indent=2), flush=True)

(out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
