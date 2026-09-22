import subprocess
import sys
from pathlib import Path

import numpy as np


def _make_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.array(rng.integers(0, 256, 8192), dtype=np.uint16).tofile(data_dir / "train.bin")
    np.array(rng.integers(0, 256, 2048), dtype=np.uint16).tofile(data_dir / "val.bin")


def test_smoke_train_and_resume(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    _make_data(data_dir)

    base = [
        sys.executable, "train.py",
        "--config", "configs/smoke.yaml",
        "--data-dir", str(data_dir),
        "--out-dir", str(out_dir),
    ]
    r = subprocess.run(base, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert (out_dir / "ckpt.pt").exists()
    assert (out_dir / "train_log.csv").exists()

    # Resume for 10 more steps: total 20
    r = subprocess.run(
        base + ["--resume", "--max-steps", "20"], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, r.stderr
    import torch

    ckpt = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ckpt["step"] == 20
    assert ckpt["tokens"] == 20 * 4 * 2 * 64  # steps * micro_batch * grad_accum * context_len


def test_smoke_train_muon(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    _make_data(data_dir)

    r = subprocess.run(
        [
            sys.executable, "train.py",
            "--config", "configs/smoke-muon.yaml",
            "--data-dir", str(data_dir),
            "--out-dir", str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr
    import torch

    ckpt = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ckpt["step"] == 6
    assert set(ckpt["optimizer"]) == {"muon", "adam"}
