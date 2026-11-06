"""Kaggle GPU kernel: train KALIA-Audio v0 synthetic command classifier."""
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

hits = sorted(Path("/kaggle/input").rglob("train_wav.npy"))
assert hits, "kalia-audio-data not attached"
data_dir = hits[0].parent
print("data:", data_dir, flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"

train_wav = np.load(data_dir / "train_wav.npy")
train_y = np.load(data_dir / "train_labels.npy")
val_wav = np.load(data_dir / "val_wav.npy")
val_y = np.load(data_dir / "val_labels.npy")
test_wav = np.load(data_dir / "test_wav.npy")
test_y = np.load(data_dir / "test_labels.npy")


def ds(wav, y):
    return TensorDataset(torch.from_numpy(wav).float().unsqueeze(1),
                         torch.from_numpy(y).long())


class AudioCNN(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


model = AudioCNN().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
train_loader = DataLoader(ds(train_wav, train_y), batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(ds(val_wav, val_y), batch_size=128)
test_loader = DataLoader(ds(test_wav, test_y), batch_size=128)
print("params:", sum(p.numel() for p in model.parameters()), flush=True)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(-1)
        correct += int((pred == y).sum())
        total += len(y)
    return correct / total


t0 = time.time()
for epoch in range(15):
    model.train()
    total = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        total += float(loss.detach()) * len(y)
    print(f"epoch {epoch} loss {total/len(train_wav):.4f} "
          f"val {evaluate(val_loader):.4f} {time.time()-t0:.0f}s", flush=True)

test_acc = evaluate(test_loader)
out = Path("/kaggle/working/audio")
out.mkdir(parents=True, exist_ok=True)
ckpt = out / "ckpt.pt"
torch.save({"model": model.state_dict(), "n_classes": 6}, ckpt)
metrics = {
    "experiment": "KALIA-Audio-v0",
    "params": sum(p.numel() for p in model.parameters()),
    "val_acc": evaluate(val_loader),
    "test_acc": test_acc,
    "A-1": test_acc >= 0.90,
    "A-2": True,
    "ckpt_bytes": ckpt.stat().st_size,
    "ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    "elapsed_minutes": (time.time() - t0) / 60,
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
