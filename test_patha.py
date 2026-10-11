"""Tests for X29 patha replay orderings.

The patterns are transcribed from the pāṭha descriptions and pinned exactly:
krama links overlapping pairs forward (12, 23, 34), jata recites each pair
forward-reverse-forward (12, 21, 12), ghana nests triples (12, 21, 123, 321,
123). An off-by-one in these sequences would silently make the arm a
different experiment, so the tests assert the recited units themselves.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from data import TokenDataset
from patha import PATTERNS, PathaReplayDataset, patha_stream

ROOT = Path(__file__).resolve().parent


def test_krama_links_overlapping_pairs_forward():
    assert patha_stream(4, "krama") == [0, 1, 1, 2, 2, 3]


def test_jata_recites_each_pair_forward_reverse_forward():
    # 12, 21, 12 then 23, 32, 23
    assert patha_stream(3, "jata") == [0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2]


def test_ghana_nests_triples():
    # 12, 21, 123, 321, 123 then 23, 32, 234, 432, 234
    assert patha_stream(4, "ghana") == [
        0, 1, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2,
        1, 2, 2, 1, 1, 2, 3, 3, 2, 1, 1, 2, 3,
    ]


def test_uniform_is_a_deterministic_permutation():
    a = patha_stream(16, "uniform", seed=3)
    b = patha_stream(16, "uniform", seed=3)
    assert a == b
    assert sorted(a) == list(range(16))


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        patha_stream(4, "nope")
    with pytest.raises(ValueError):
        patha_stream(1, "jata")
    with pytest.raises(ValueError):
        patha_stream(2, "ghana")
    with pytest.raises(ValueError):
        patha_stream(0, "uniform")


def _ds(tmp_path: Path, name: str, seed: int, tokens: int = 4000, ctx: int = 32) -> TokenDataset:
    rng = np.random.default_rng(seed)
    p = tmp_path / f"{name}.bin"
    rng.integers(0, 512, size=tokens).astype(np.uint16).tofile(p)
    return TokenDataset(p, context_len=ctx)


def test_replay_rows_follow_the_pattern_over_the_fixed_base_set(tmp_path):
    new = _ds(tmp_path, "new", 1)
    replay = _ds(tmp_path, "replay", 2)
    ds = PathaReplayDataset(
        new, replay, replay_prob=0.5, pattern="krama", base_windows=8, seed=5
    )
    x, y = ds._replay_rows(3)
    # krama stream starts [0, 1, 1, ...]: first two rows are base windows 0 and 1
    for row, index in ((0, 0), (1, 1)):
        start = ds._base_offsets[index]
        expected = replay.tokens[start : start + replay.context_len].astype(np.int64)
        assert torch.equal(x[row], torch.tensor(expected))
        assert torch.equal(y[row][:-1], x[row][1:]), "targets must be the true successors"


def test_cursor_cycles_through_the_pattern(tmp_path):
    new = _ds(tmp_path, "new", 1)
    replay = _ds(tmp_path, "replay", 2)
    ds = PathaReplayDataset(new, replay, pattern="jata", base_windows=4, seed=0)
    cycle = len(ds._stream)
    first_x, _ = ds._replay_rows(cycle)
    wrapped_x, _ = ds._replay_rows(1)
    assert torch.equal(first_x[0], wrapped_x[0]), "the pattern must cycle"


def test_same_seed_reproduces_the_same_stream(tmp_path):
    new = _ds(tmp_path, "new", 1)
    replay = _ds(tmp_path, "replay", 2)
    a = PathaReplayDataset(new, replay, pattern="ghana", base_windows=8, seed=11)
    b = PathaReplayDataset(new, replay, pattern="ghana", base_windows=8, seed=11)
    assert a._stream == b._stream
    assert a._base_offsets == b._base_offsets
    xa, _ = a._replay_rows(5)
    xb, _ = b._replay_rows(5)
    assert torch.equal(xa, xb)


def test_batches_carry_new_and_replay_rows_with_masks(tmp_path):
    new = _ds(tmp_path, "new", 1)
    replay = _ds(tmp_path, "replay", 2)
    ds = PathaReplayDataset(new, replay, replay_prob=0.5, pattern="ghana", seed=0)
    gen = torch.Generator().manual_seed(0)
    x, y, attn, loss_mask = ds.get_batch(8, torch.device("cpu"), gen, return_masks=True)
    assert x.shape == (8, 32) and y.shape == (8, 32)
    assert attn.shape == (8, 1, 32, 32)
    assert loss_mask.shape == (8, 32)
    assert ds.context_len == 32 and len(ds) == len(new)


def test_budget_and_pattern_validation(tmp_path):
    new = _ds(tmp_path, "new", 1)
    replay = _ds(tmp_path, "replay", 2)
    with pytest.raises(ValueError):
        PathaReplayDataset(new, replay, replay_prob=1.0)
    with pytest.raises(ValueError):
        PathaReplayDataset(new, replay, pattern="mystery")
    assert PATTERNS == ("uniform", "krama", "jata", "ghana")


def test_train_py_runs_with_a_patha_pattern(tmp_path):
    """The wiring is a separate claim from the sampler: an unexercised config
    path is this project's characteristic defect class (D51, I19, I20)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    torch.randint(0, 256, (40_000,), generator=torch.Generator().manual_seed(1)).to(
        torch.uint16
    ).numpy().tofile(data_dir / "train.bin")
    torch.randint(0, 256, (8_000,), generator=torch.Generator().manual_seed(2)).to(
        torch.uint16
    ).numpy().tofile(data_dir / "val.bin")
    torch.randint(0, 256, (20_000,), generator=torch.Generator().manual_seed(3)).to(
        torch.uint16
    ).numpy().tofile(data_dir / "replay.bin")
    cfg = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    cfg["train"].update({"replay_prob": 0.5, "replay_pattern": "jata", "eval_interval": 5})
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--config", str(cfg_path),
         "--data-dir", str(data_dir), "--out-dir", str(out),
         "--max-steps", "10", "--seed", "7", "--replay-bin", str(data_dir / "replay.bin")],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "pattern: jata" in r.stdout
    assert (out / "train_log.csv").exists()
    assert (out / "ckpt.pt").exists()
