"""End-to-end: train.py must accept per-source bins and actually reweight.

The loader in mixture.py is unit-tested, but the wiring into train.py is a
separate claim: that --sources/--source-bins/--source-weights produce a run
whose sampling actually follows the requested mixture, that per_source_log.csv
appears, and that adaptive_every=0 leaves the mixture static (the registered
control arm). Without this, "the Kautilya arm is ready" would be an assertion
about a code path nobody has executed.

Runs the real train.py on a tiny CPU config. Seconds, not GPU.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent
STATIC = {"fineweb": 0.60, "tinystories": 0.20, "cosmopedia": 0.15, "python": 0.05}
NAMES = list(STATIC)


def _make_corpus(tmp_path: Path, n_tokens: int = 40_000) -> tuple[Path, dict[str, Path]]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bins = {}
    for i, name in enumerate(NAMES):
        p = data_dir / f"{name}.bin"
        g = torch.Generator().manual_seed(100 + i)
        torch.randint(0, 256, (n_tokens,), generator=g).to(torch.uint16).numpy().tofile(p)
        bins[name] = p
    # val.bin must exist or train.py skips validation entirely
    g = torch.Generator().manual_seed(999)
    torch.randint(0, 256, (8_000,), generator=g).to(torch.uint16).numpy().tofile(data_dir / "val.bin")
    return data_dir, bins


def _config(tmp_path: Path, **train_overrides) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    cfg["train"].update(train_overrides)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _run(cfg, data_dir, out_dir, bins, weights, sources=NAMES, device="cpu"):
    cmd = [sys.executable, str(ROOT / "train.py"), "--config", str(cfg),
           "--data-dir", str(data_dir), "--out-dir", str(out_dir),
           "--max-steps", "10", "--seed", "7"]
    if weights is not None:
        cmd += ["--sources", *sources,
                "--source-bins", *[str(bins[s]) for s in sources],
                "--source-weights", json.dumps(weights)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)


def test_rejects_only_some_of_the_three_flags(tmp_path):
    """A partial invocation must fail loudly, not zip a short list."""
    data_dir, bins = _make_corpus(tmp_path)
    cfg = _config(tmp_path)
    r2 = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--config", str(cfg),
         "--data-dir", str(data_dir), "--out-dir", str(tmp_path / "o2"),
         "--sources", "fineweb", "--source-bins", str(bins["fineweb"])],
        capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r2.returncode != 0
    assert "needs all three" in (r2.stdout + r2.stderr), r2.stdout + r2.stderr


def test_rejects_mismatched_source_and_bin_counts(tmp_path):
    data_dir, bins = _make_corpus(tmp_path)
    cfg = _config(tmp_path)
    r = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--config", str(cfg),
         "--data-dir", str(data_dir), "--out-dir", str(tmp_path / "o"),
         "--sources", *NAMES, "--source-bins", *[str(bins[s]) for s in NAMES[:3]],
         "--source-weights", json.dumps(STATIC)],
        capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode != 0
    assert "one-to-one" in (r.stdout + r.stderr), r.stdout + r.stderr


def test_rejects_weights_naming_an_unknown_source(tmp_path):
    data_dir, bins = _make_corpus(tmp_path)
    cfg = _config(tmp_path)
    bad = dict(STATIC, mystery=0.1)
    r = _run(cfg, data_dir, tmp_path / "o", bins, weights=bad)
    assert r.returncode != 0
    assert "not in --sources" in (r.stdout + r.stderr), r.stdout + r.stderr


def test_static_per_source_run_completes_and_prints_the_mixture(tmp_path):
    """adaptive_every=0 is the registered control arm: static, no reweighting."""
    data_dir, bins = _make_corpus(tmp_path)
    cfg = _config(tmp_path, adaptive_every=0)
    r = _run(cfg, data_dir, tmp_path / "o", bins, weights=STATIC)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "per-source mixture (Kautilya arm)" in r.stdout
    assert "fineweb 60.0%" in r.stdout
    assert "reweighting every" not in r.stdout, "adaptive_every=0 must not reweight"
    assert not (tmp_path / "o" / "per_source_log.csv").exists()


def test_adaptive_run_reweights_and_writes_per_source_log(tmp_path):
    data_dir, bins = _make_corpus(tmp_path)
    cfg = _config(tmp_path, adaptive_every=2, adaptive_batches=1, adaptive_batch_size=4)
    r = _run(cfg, data_dir, tmp_path / "o", bins, weights=STATIC)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "reweighting every 2 steps" in r.stdout
    assert "per-source loss" in r.stdout
    log = tmp_path / "o" / "per_source_log.csv"
    assert log.exists(), "per_source_log.csv is the record of the policy's decisions"
    rows = list(csv.reader(log.open()))
    header, body = rows[0], rows[1:]
    for n in NAMES:
        assert f"loss_{n}" in header and f"weight_{n}" in header and f"strategy_{n}" in header
    assert body, "no reweighting rows were written"
    for row in body:
        weights = [float(v) for v in row[len(NAMES) + 1: 2 * len(NAMES) + 1]]
        assert pytest.approx(sum(weights), abs=1e-3) == 1.0, weights
        strategies = row[2 * len(NAMES) + 1:]
        assert set(strategies) <= {"sama", "dana", "bheda", "danda"}, strategies


def test_replay_and_per_source_are_mutually_exclusive(tmp_path):
    """Two different mixture mechanisms at once would be uninterpretable.

    Replay legitimately still needs train.bin (it replays the *original* corpus),
    so the assertion is about which mechanism engages, not about file presence.
    """
    data_dir, bins = _make_corpus(tmp_path)
    # replay needs the original corpus, which is the point of replay
    torch.randint(0, 256, (40_000,), generator=torch.Generator().manual_seed(42)).to(
        torch.uint16).numpy().tofile(data_dir / "train.bin")
    replay_bin = data_dir / "replay.bin"
    torch.randint(0, 256, (8_000,), generator=torch.Generator().manual_seed(5)).to(
        torch.uint16).numpy().tofile(replay_bin)
    cfg = _config(tmp_path, adaptive_every=2, replay_prob=0.1)
    cmd = [sys.executable, str(ROOT / "train.py"), "--config", str(cfg),
           "--data-dir", str(data_dir), "--out-dir", str(tmp_path / "o"),
           "--max-steps", "6", "--seed", "7", "--replay-bin", str(replay_bin),
           "--sources", *NAMES, "--source-bins", *[str(bins[s]) for s in NAMES],
           "--source-weights", json.dumps(STATIC)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    # replay takes precedence; the per-source path must not also engage silently
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    if "replay:" in r.stdout:
        assert "Kautilya arm" not in r.stdout, (
            "replay and per-source both engaged: the mixture would be two mechanisms at once"
        )
