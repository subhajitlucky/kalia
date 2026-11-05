"""Kaggle GPU kernel: one X48 tokenizer-A/B arm.

Set TOKENIZER and SEED at the top. The kernel trains one 30M arm on the
pre-tokenized matched data and saves the checkpoint. Evaluation is a separate
CPU kernel so the GPU session stays inside the 9-hour limit.

Arms: gpt2/1338, gpt2/1339, ktok/1338, ktok/1339.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TOKENIZER = "gpt2"
SEED = 1338

assert TOKENIZER in ("gpt2", "ktok")
assert SEED in (1338, 1339)

hits = sorted(Path("/kaggle/input").rglob("train.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))
print("code:", code_root, flush=True)

train_name = f"train_{TOKENIZER}.bin"
val_name = f"val_{TOKENIZER}.bin"
data_hits = sorted(Path("/kaggle/input").rglob(train_name))
assert data_hits, f"{train_name} missing; attach kalia-x48-data"
train_bin = data_hits[0]
val_bin = train_bin.parent / val_name
assert val_bin.exists(), val_bin
print("data:", train_bin, val_bin, flush=True)

data_dir = Path(f"/kaggle/working/data_{TOKENIZER}")
data_dir.mkdir(parents=True, exist_ok=True)
for name, src in (("train.bin", train_bin), ("val.bin", val_bin)):
    dst = data_dir / name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)

config = work / "configs" / f"x48-{TOKENIZER}.yaml"
assert config.exists(), config
out_dir = Path(f"/kaggle/working/x48/{TOKENIZER}_{SEED}")
out_dir.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable, "train.py",
    "--config", str(config),
    "--data-dir", str(data_dir),
    "--out-dir", str(out_dir),
    "--seed", str(SEED),
]
print("cmd:", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

ckpt = out_dir / "ckpt.pt"
assert ckpt.exists(), ckpt
sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
meta = {
    "experiment": "X48-arm",
    "tokenizer": TOKENIZER,
    "seed": SEED,
    "config": str(config),
    "train_bin": str(train_bin),
    "val_bin": str(val_bin),
    "ckpt": str(ckpt),
    "ckpt_bytes": ckpt.stat().st_size,
    "ckpt_sha256": sha,
}
(out_dir / "arm_manifest.json").write_text(
    json.dumps(meta, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
