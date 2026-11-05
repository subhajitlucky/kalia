"""Kaggle CPU kernel: X48 matched data for the tokenizer A/B.

Takes the first 312M GPT-2 tokens of the v2b training shard, copies them for
the GPT-2 arm, decodes the same tokens to text, and re-tokenizes that text with
K-Tok 32k. The two arms therefore see the same underlying text.

CPU only. Outputs uint16 .bin shards + manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

N_TRAIN = 312_000_000
N_VAL = 10_000_000

train_hits = sorted(Path("/kaggle/input").rglob("train.bin"))
preferred = [p for p in train_hits if "kalia-prep-v2b" in str(p)]
assert preferred, f"kalia-prep-v2b not attached: {train_hits}"
train_bin = preferred[0]
val_bin = train_bin.parent / "val.bin"
assert val_bin.exists()

tok_hits = sorted(Path("/kaggle/input").rglob("ktok_32k.json"))
assert tok_hits, "kalia-k-tok-v0 output not attached"
ktok_path = tok_hits[0]
print("train:", train_bin, "val:", val_bin, "ktok:", ktok_path, flush=True)

try:
    import tiktoken
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"],
                   check=True)
    import tiktoken
try:
    from tokenizers import Tokenizer
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"],
                   check=True)
    from tokenizers import Tokenizer

gpt2 = tiktoken.get_encoding("gpt2")
ktok = Tokenizer.from_file(str(ktok_path))
out = Path("/kaggle/working/x48-data")
out.mkdir(parents=True, exist_ok=True)


def copy_tokens(src: Path, n: int, dst: Path) -> int:
    arr = np.memmap(src, dtype=np.uint16, mode="r")
    n = min(n, len(arr))
    with open(dst, "wb") as fh:
        for start in range(0, n, 10_000_000):
            chunk = np.asarray(arr[start:min(start + 10_000_000, n)], dtype=np.uint16)
            chunk.tofile(fh)
    return n


def retokenize(src: Path, n: int, dst: Path) -> int:
    arr = np.memmap(src, dtype=np.uint16, mode="r")
    n = min(n, len(arr))
    written = 0
    with open(dst, "wb") as fh:
        for start in range(0, n, 1_000_000):
            chunk = np.asarray(arr[start:min(start + 1_000_000, n)], dtype=np.int64)
            text = gpt2.decode(chunk.tolist())
            ids = ktok.encode(text).ids
            np.array(ids, dtype=np.uint16).tofile(fh)
            written += len(ids)
            if start % 50_000_000 == 0:
                print(f"retokenized {start:,}/{n:,} gpt2 tokens -> {written:,} ktok",
                      flush=True)
    return written


manifest = {
    "experiment": "X48-data",
    "source_train": str(train_bin),
    "source_val": str(val_bin),
    "ktok": str(ktok_path),
    "requested_train_gpt2": N_TRAIN,
    "requested_val_gpt2": N_VAL,
}
print("copying GPT-2 arm", flush=True)
manifest["train_gpt2_tokens"] = copy_tokens(train_bin, N_TRAIN, out / "train_gpt2.bin")
manifest["val_gpt2_tokens"] = copy_tokens(val_bin, N_VAL, out / "val_gpt2.bin")
print("GPT-2:", manifest["train_gpt2_tokens"], manifest["val_gpt2_tokens"], flush=True)

print("re-tokenizing K-Tok arm", flush=True)
manifest["train_ktok_tokens"] = retokenize(train_bin, N_TRAIN, out / "train_ktok.bin")
manifest["val_ktok_tokens"] = retokenize(val_bin, N_VAL, out / "val_ktok.bin")
print("K-Tok:", manifest["train_ktok_tokens"], manifest["val_ktok_tokens"], flush=True)

for name in ("train_gpt2.bin", "val_gpt2.bin", "train_ktok.bin", "val_ktok.bin"):
    p = out / name
    manifest[name] = {
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
