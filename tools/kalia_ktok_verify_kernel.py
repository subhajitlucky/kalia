"""Kaggle CPU kernel: verify K-Tok v0 round-trip losslessness.

The first report used tokenizer.decode() with its default
skip_special_tokens=True, which drops <|endoftext|> and makes every round-trip
look false. This kernel reloads the saved tokenizers and checks with
skip_special_tokens=False on held-out text and on an explicit special-token
sample.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from tokenizers import Tokenizer
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"],
                   check=True)
    from tokenizers import Tokenizer

hits = sorted(Path("/kaggle/input").rglob("ktok_32k.json"))
assert hits, "K-Tok output not attached"
root = hits[0].parent
val_path = root / "val.txt"
assert val_path.exists(), f"val.txt missing in {root}"
text = val_path.read_text()[:200_000]
special = "<|endoftext|> Bella gave Tara the shiny book. <|endoftext|>"

report = {"experiment": "K-Tok-v0-roundtrip", "root": str(root), "samples": {}}
for name in ("ktok_32k", "ktok_32k_digits", "ktok_50k", "ktok_64k"):
    path = root / f"{name}.json"
    assert path.exists(), path
    tok = Tokenizer.from_file(str(path))
    row = {}
    for label, sample in (("heldout", text), ("special", special)):
        ids = tok.encode(sample).ids
        decoded = tok.decode(ids, skip_special_tokens=False)
        nbytes = len(sample.encode("utf-8"))
        row[label] = {
            "bytes": nbytes,
            "tokens": len(ids),
            "bytes_per_token": nbytes / max(1, len(ids)),
            "roundtrip": decoded == sample,
        }
    report["samples"][name] = row
    print(name, json.dumps(row), flush=True)

Path("/kaggle/working/ktok_verify.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print("wrote /kaggle/working/ktok_verify.json")
