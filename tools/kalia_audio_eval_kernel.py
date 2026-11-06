"""Kaggle CPU kernel: per-class audio command check."""
from __future__ import annotations
import json, os, shutil, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

hits = sorted(Path("/kaggle/input").rglob("test_wav.npy"))
assert hits, "audio data not attached"
data_dir = hits[0].parent
ckpts = sorted(Path("/kaggle/input").rglob("ckpt.pt"))
ckpts = [p for p in ckpts if "audio" in str(p)]
assert ckpts, "audio checkpoint not attached"
ckpt = ckpts[0]

class AudioCNN(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, n_classes)
    def forward(self, x):
        return self.fc(self.net(x).flatten(1))

model = AudioCNN()
model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
model.eval()
wav = torch.from_numpy(np.load(data_dir / "test_wav.npy")).float().unsqueeze(1)
y = torch.from_numpy(np.load(data_dir / "test_labels.npy")).long()
with torch.no_grad():
    pred = model(wav).argmax(-1)
per = {}
for c in range(6):
    mask = y == c
    per[str(c)] = {"n": int(mask.sum()), "acc": float((pred[mask] == y[mask]).float().mean())}
out = {"test_acc": float((pred == y).float().mean()), "per_class": per,
       "all_classes_predicted": sorted(set(pred.tolist()))}
Path("/kaggle/working/audio_eval.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
