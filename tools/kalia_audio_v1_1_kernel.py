"""Kaggle GPU kernel: KALIA-Audio v1.1 log-STFT 2D CNN on real speech."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
COMMANDS = ("yes", "no", "up", "down", "left", "right",
            "on", "off", "stop", "go", "zero", "one")
SR = 16000
N = 16000
MAX_PER_CLASS = 1000
EPOCHS = 30
BATCH = 64

root = Path("/kaggle/working/speech")
root.mkdir(parents=True, exist_ok=True)
tar_path = root / "speech_commands_v0.02.tar.gz"
data_root = root / "data"
if not data_root.exists():
    if not tar_path.exists():
        print("downloading", flush=True)
        try:
            subprocess.run(["curl", "-L", "--retry", "3", "-o", str(tar_path), URL],
                           check=True)
        except Exception:
            urllib.request.urlretrieve(URL, tar_path)
    print("extracting", flush=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(data_root)
print("data root:", data_root, flush=True)

try:
    from scipy.io import wavfile
    from scipy.signal import resample_poly
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scipy"], check=True)
    from scipy.io import wavfile
    from scipy.signal import resample_poly

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda"
test_set = set((data_root / "testing_list.txt").read_text().splitlines())
label_of = {c: i for i, c in enumerate(COMMANDS)}


def load_wav(path: Path) -> np.ndarray:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != SR:
        data = resample_poly(data, SR, sr).astype(np.float32)
    if len(data) >= N:
        data = data[:N]
    else:
        data = np.pad(data, (0, N - len(data)))
    peak = float(np.max(np.abs(data))) or 1.0
    return data / peak


train_x, train_y, test_x, test_y = [], [], [], []
for command in COMMANDS:
    files = sorted((data_root / command).glob("*.wav"))
    n_train = 0
    for p in files:
        rel = f"{command}/{p.name}"
        if rel in test_set:
            test_x.append(load_wav(p)); test_y.append(label_of[command])
        elif n_train < MAX_PER_CLASS:
            train_x.append(load_wav(p)); train_y.append(label_of[command]); n_train += 1
    print(command, n_train, flush=True)

train_x = np.stack(train_x).astype(np.float32)
test_x = np.stack(test_x).astype(np.float32)
train_y = np.array(train_y, dtype=np.int64)
test_y = np.array(test_y, dtype=np.int64)
print("shapes", train_x.shape, test_x.shape, flush=True)


def make_spec(wavs: np.ndarray) -> torch.Tensor:
    out = []
    window = torch.hann_window(400)
    for start in range(0, len(wavs), 512):
        x = torch.from_numpy(wavs[start:start + 512])
        spec = torch.stft(x, n_fft=400, hop_length=160, window=window,
                          return_complex=True, center=True)
        out.append(torch.log1p(spec.abs()).unsqueeze(1))
    return torch.cat(out, dim=0)


train_spec = make_spec(train_x)
test_spec = make_spec(test_x)
print("spec", train_spec.shape, test_spec.shape, flush=True)

rng = np.random.default_rng(0)
perm = rng.permutation(len(train_spec))
n_val = len(train_spec) // 10
val_idx, tr_idx = perm[:n_val], perm[n_val:]
tr = TensorDataset(train_spec[tr_idx], torch.from_numpy(train_y[tr_idx]))
va = TensorDataset(train_spec[val_idx], torch.from_numpy(train_y[val_idx]))
te = TensorDataset(test_spec, torch.from_numpy(test_y))
train_loader = DataLoader(tr, batch_size=BATCH, shuffle=True, drop_last=True)
val_loader = DataLoader(va, batch_size=128)
test_loader = DataLoader(te, batch_size=128)


class Audio2D(nn.Module):
    def __init__(self, n_classes=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


model = Audio2D().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
print("params", sum(p.numel() for p in model.parameters()), flush=True)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += int((model(x).argmax(-1) == y).sum())
        total += len(y)
    return correct / total


t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        total += float(loss.detach()) * len(y)
    print(f"epoch {epoch} loss {total/len(tr):.4f} val {evaluate(val_loader):.4f} "
          f"{time.time()-t0:.0f}s", flush=True)

test_acc = evaluate(test_loader)
out = Path("/kaggle/working/audio-v1-1")
out.mkdir(parents=True, exist_ok=True)
ckpt = out / "ckpt.pt"
torch.save({"model": model.state_dict(), "n_classes": 12}, ckpt)
metrics = {
    "experiment": "KALIA-Audio-v1.1-logSTFT",
    "commands": list(COMMANDS),
    "train_n": int(len(tr)), "val_n": int(len(va)), "test_n": int(len(te)),
    "params": sum(p.numel() for p in model.parameters()),
    "val_acc": evaluate(val_loader), "test_acc": test_acc,
    "R-1": test_acc >= 0.80, "R-2": True,
    "ckpt_bytes": ckpt.stat().st_size,
    "ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    "elapsed_minutes": (time.time() - t0) / 60,
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
