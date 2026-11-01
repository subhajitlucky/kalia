"""Kaggle CPU kernel: materialise the v0.3.1 w0.3 merged checkpoint.

The v0.3.1 merge kernel scored the 30% v0.2.0 / 70% v0.3.0 blend at val
2.7579, all five 2k tasks positive, CL-0 2.740949. Its cleanup then deleted the
best checkpoint after copying it (the same dict was updated and unlinked). The
numbers are valid; the binary is gone.

This kernel rebuilds the exact blend from the two hash-pinned source
checkpoints, re-scores the 100-batch val on CPU to confirm 2.7579, and outputs
``best_merged.pt``.

CPU only.
"""

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

KERNEL_ID = "unknown"
KERNEL_REVISION = "unknown"
WEIGHT_V020 = 0.3
EXPECTED_VAL = 2.7579


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def pick(pattern: str, markers: tuple[str, ...]) -> list[str]:
    found = sorted(glob.glob(pattern, recursive=True))
    preferred = [p for p in found if any(m in p for m in markers)]
    return preferred or found


# code, for the val re-score
hits = pick("/kaggle/input/**/eval_val.py", ("/kalia-code-dev/",))
assert hits, "attach kalia-code-dev"
code_root = os.path.dirname(hits[0])
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("eval_val.py", "model.py", "data.py", "eval_reversibility.py", "kalia_lm.py"):
    assert os.path.exists(req), req

v020_hits = pick("/kaggle/input/**/kalia-v020-ckpt/ckpt.pt", ("/kalia-v020-ckpt/",))
assert v020_hits, "attach kalia-v020-ckpt"
v030_hits = pick("/kaggle/input/**/out/ckpt.pt", ("/kalia-v030-epoch2/",))
assert v030_hits, "attach kalia-v030-epoch2"
val_hits = pick("/kaggle/input/**/val.bin", ("/kalia-prep-v2b/",))
assert val_hits, "attach kalia-prep-v2b"
V020, V030, VAL = v020_hits[0], v030_hits[0], val_hits[0]
print("v0.2.0:", V020)
print("v0.3.0:", V030)

a = torch.load(V020, map_location="cpu", weights_only=False)
b = torch.load(V030, map_location="cpu", weights_only=False)
for key in ("n_layer", "n_head", "n_embd", "context_len", "vocab_size"):
    assert a["config"]["model"][key] == b["config"]["model"][key], key
for key, default in (("qk_norm", False), ("logit_softcap", 0.0), ("nope_interval", 0),
                     ("gated_residual", False), ("branch_norm", False)):
    assert a["config"]["model"].get(key, default) == b["config"]["model"].get(key, default), key
assert set(a["model"]) == set(b["model"])
for key in a["model"]:
    assert a["model"][key].shape == b["model"][key].shape, key

state = {}
for key in b["model"]:
    merged = (1.0 - WEIGHT_V020) * b["model"][key].float() + WEIGHT_V020 * a["model"][key].float()
    state[key] = merged.to(b["model"][key].dtype)
ckpt = {
    "model": state,
    "optimizer": b["optimizer"],
    "step": b["step"],
    "tokens": b["tokens"],
    "config": b["config"],
    "merged_from": [V020, V030],
    "merge_weights": [WEIGHT_V020, 1.0 - WEIGHT_V020],
}
out = Path("/kaggle/working/best_merged.pt")
torch.save(ckpt, out)
print("wrote", out, out.stat().st_size)

# weight checksum independent of the zip serialisation
h = hashlib.sha256()
for key in sorted(state):
    h.update(key.encode())
    h.update(state[key].float().numpy().tobytes())
weight_sha = h.hexdigest()
print("weight sha256:", weight_sha)

subprocess.run(
    [sys.executable, "eval_val.py", "--ckpt", str(out), "--val-bin", VAL,
     "--batches", "100", "--batch-size", "8", "--seed", "1234",
     "--out", "/kaggle/working/best_merged_val.md", "--device", "cpu"],
    check=True,
)
text = Path("/kaggle/working/best_merged_val.md").read_text()
val = float(re.search(r"Val loss:\s*([0-9.]+)", text).group(1))
print(f"val {val:.4f} (expected {EXPECTED_VAL:.4f})")
assert abs(val - EXPECTED_VAL) <= 0.001, (val, EXPECTED_VAL)

manifest = {
    "kernel_id": KERNEL_ID,
    "kernel_revision": KERNEL_REVISION,
    "recorded": datetime.now(timezone.utc).isoformat(),
    "weights": {"v020": WEIGHT_V020, "v030": 1.0 - WEIGHT_V020},
    "sources": {
        "v020": {"path": V020, "sha256": sha256_file(V020)},
        "v030": {"path": V030, "sha256": sha256_file(V030)},
    },
    "val_bin": {"path": VAL, "sha256": sha256_file(VAL)},
    "checkpoint": {"path": str(out), "sha256": sha256_file(out),
                   "bytes": out.stat().st_size, "weight_sha256": weight_sha},
    "val": val,
    "expected_val": EXPECTED_VAL,
}
write_json("/kaggle/working/best_merged_manifest.json", manifest)
print(json.dumps(manifest, indent=2, sort_keys=True))
