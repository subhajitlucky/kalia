"""Kaggle CPU kernel: evaluate both X49 checkpoints on the shared control val."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import torch
from eval_reversibility import load_model

hits = sorted(Path("/kaggle/input").rglob("ckpt.pt"))
ctrl = [p for p in hits if "x49-control" in str(p)]
treat = [p for p in hits if "x49-treatment" in str(p)]
assert ctrl and treat, hits
val_hits = sorted(Path("/kaggle/input").rglob("val.bin"))
val_hits = [p for p in val_hits if "x49-data/control" in str(p)]
assert val_hits, "control val.bin missing"
val_path = val_hits[0]
print("control:", ctrl[0], "treatment:", treat[0], "val:", val_path, flush=True)

tokens = np.memmap(val_path, dtype=np.uint16, mode="r")
context = 1024
batch = 8
batches = 100
rng = np.random.default_rng(1234)
starts = rng.integers(0, len(tokens) - context - 1, size=(batches, batch))
x = torch.stack([torch.from_numpy(tokens[s:s+context].astype(np.int64)) for row in starts for s in row])
y = torch.stack([torch.from_numpy(tokens[s+1:s+1+context].astype(np.int64)) for row in starts for s in row])
print("eval tokens", x.numel(), flush=True)


def evaluate(path):
    model = load_model(path, torch.device("cpu"))
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(x), 32):
            _, loss = model(x[i:i+32], y[i:i+32])
            total += float(loss) * len(x[i:i+32])
            n += len(x[i:i+32])
    return total / n

out = {"val_tokens": int(x.numel()), "control": evaluate(ctrl[0]), "treatment": evaluate(treat[0])}
out["delta"] = out["treatment"] - out["control"]
out["K-4"] = out["treatment"] <= out["control"] + 0.01
Path("/kaggle/working/x49_val.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
