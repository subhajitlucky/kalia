"""X40: the train.py wiring, exercised end to end on CPU.

A model class that passes unit tests but is never reached by the training loop
is this project's characteristic defect. This test runs the real entrypoint
with the pair batch, the event workspace and the auxiliary loss for one step.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import yaml

from intervention_pairs import write_pairs

ROOT = Path(__file__).resolve().parent


def _data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    torch.randint(0, 256, (20_000,), generator=torch.Generator().manual_seed(1)).to(
        torch.uint16
    ).numpy().tofile(data / "train.bin")
    torch.randint(0, 256, (4_000,), generator=torch.Generator().manual_seed(2)).to(
        torch.uint16
    ).numpy().tofile(data / "val.bin")
    return data


def _config(tmp_path: Path, **model_extra) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    cfg["model"].update({
        "vocab_size": 50257,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 64,
        "context_len": 128,
        "event_slots": 4,
        "event_slot_dim": 8,
        "event_chunk": 64,
        "event_head": True,
    })
    cfg["model"].update(model_extra)
    cfg["train"].update({
        "micro_batch_size": 2,
        "grad_accum_steps": 1,
        "max_steps": 1,
        "warmup_steps": 1,
        "eval_interval": 0,
        "log_interval": 1,
        "ip_lm_weight": 1.0,
        "ip_state_weight": 0.1,
        "ip_div_weight": 0.01,
        "ip_batch_size": 2,
        "ip_prefix_len": 64,
        "ip_suffix_len": 64,
    })
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_train_py_runs_the_ip_path(tmp_path):
    data = _data(tmp_path)
    pairs = tmp_path / "pairs"
    write_pairs(pairs, seed=7, n_train=10, n_eval=5, n_comp=5)
    cfg = _config(tmp_path)
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "train.py"),
            "--config", str(cfg),
            "--data-dir", str(data),
            "--out-dir", str(out),
            "--ip-pairs", str(pairs / "train.jsonl"),
            "--max-steps", "1",
            "--seed", "7",
        ],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ip: pairs" in result.stdout
    assert "ip_state" in result.stdout
    assert (out / "ckpt.pt").exists()


def test_state_loss_requires_the_head(tmp_path):
    data = _data(tmp_path)
    pairs = tmp_path / "pairs"
    write_pairs(pairs, seed=7, n_train=10, n_eval=5, n_comp=5)
    cfg = _config(tmp_path, event_head=False)
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "train.py"),
            "--config", str(cfg),
            "--data-dir", str(data),
            "--out-dir", str(tmp_path / "out"),
            "--ip-pairs", str(pairs / "train.jsonl"),
            "--max-steps", "1",
        ],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert result.returncode != 0
    assert "event_head" in result.stdout + result.stderr
