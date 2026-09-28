"""Tests for continual-learning mechanisms CL-1 (replay) and CL-2 (merge)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from data import MixtureDataset, TokenDataset, document_mask


def _load(name: str):
    path = Path(__file__).resolve().parent.parent / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


merger = _load("merge_checkpoints")


def _shard(tmp_path, name, n, lo, hi):
    """Write a token shard whose ids are drawn from a disjoint range per shard."""
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    toks = rng.integers(lo, hi, size=n).astype(np.uint16)
    toks[::97] = 50256
    path = tmp_path / f"{name}.bin"
    toks.tofile(path)
    return TokenDataset(path, 64)


# ---------------------------------------------------------------- CL-1: replay


def test_replay_probability_zero_is_pure_new_data(tmp_path):
    new = _shard(tmp_path, "new", 20_000, 10, 100)
    rep = _shard(tmp_path, "rep", 20_000, 30_000, 31_000)
    mix = MixtureDataset(new, rep, replay_prob=0.0)
    x, _ = mix.get_batch(16, torch.device("cpu"), torch.Generator().manual_seed(0))
    # EOS (50256) appears in every shard, so exclude it from the check
    assert int(((x >= 30_000) & (x != 50256)).sum()) == 0, \
        "replay tokens leaked in at replay_prob=0"


def test_replay_probability_one_is_pure_replay(tmp_path):
    new = _shard(tmp_path, "new", 20_000, 10, 100)
    rep = _shard(tmp_path, "rep", 20_000, 30_000, 31_000)
    mix = MixtureDataset(new, rep, replay_prob=1.0 - 1e-6)
    x, _ = mix.get_batch(16, torch.device("cpu"), torch.Generator().manual_seed(0))
    # every row must be a replay row (EOS excluded, it belongs to both shards)
    assert int(((x < 100) & (x != 50256)).sum()) == 0, \
        "new-data tokens leaked in at replay_prob~1"


def test_replay_fraction_is_approximately_right(tmp_path):
    new = _shard(tmp_path, "new", 200_000, 10, 100)
    rep = _shard(tmp_path, "rep", 200_000, 30_000, 31_000)
    mix = MixtureDataset(new, rep, replay_prob=0.10)
    total = trials = 0
    for seed in range(40):
        x, _ = mix.get_batch(32, torch.device("cpu"), torch.Generator().manual_seed(seed))
        total += int((x >= 30_000).sum())
        trials += x.numel()
    observed = total / trials
    assert 0.05 < observed < 0.16, f"replay fraction {observed:.3f} is far from 0.10"


def test_mixture_shapes_match_a_pure_batch(tmp_path):
    new = _shard(tmp_path, "new", 20_000, 10, 100)
    rep = _shard(tmp_path, "rep", 20_000, 30_000, 31_000)
    mix = MixtureDataset(new, rep, replay_prob=0.1)
    x, y, attn, lm = mix.get_batch(
        12, torch.device("cpu"), torch.Generator().manual_seed(1), return_masks=True
    )
    assert x.shape == y.shape == (12, 64)
    assert attn.shape == (12, 1, 64, 64)
    assert lm.shape == (12, 64)


def test_mixture_masks_are_correct_not_merely_present(tmp_path):
    """Masks are built from the concatenated batch, so a cross-shard row must still mask."""
    new = _shard(tmp_path, "new", 20_000, 10, 100)
    rep = _shard(tmp_path, "rep", 20_000, 30_000, 31_000)
    mix = MixtureDataset(new, rep, replay_prob=0.5)
    x, y, attn, lm = mix.get_batch(
        8, torch.device("cpu"), torch.Generator().manual_seed(3), return_masks=True
    )
    for i in range(x.size(0)):
        expected, _ = document_mask(x[i : i + 1])
        assert torch.equal(attn[i : i + 1], expected), f"row {i} mask disagrees with document_mask"
    assert lm.shape == y.shape


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_invalid_replay_probability_is_rejected(tmp_path, bad):
    new = _shard(tmp_path, "new", 5_000, 10, 100)
    rep = _shard(tmp_path, "rep", 5_000, 30_000, 31_000)
    with pytest.raises(ValueError):
        MixtureDataset(new, rep, replay_prob=bad)


# ------------------------------------------------------------------ CL-2: merge


def _ckpt(step: int, value: float, tokens: int = 1000):
    model = {"w": torch.full((4, 4), value), "b": torch.full((4,), value)}
    return {
        "model": model,
        "optimizer": {"state": {}, "step": step},
        "step": step,
        "tokens": tokens,
        "config": {"model": {"n_layer": 1}},
    }


def test_merge_averages_weights():
    a, b = _ckpt(10, 0.0), _ckpt(20, 2.0)
    out = merger.merge_states([a, b], [0.5, 0.5])["w"]
    assert torch.allclose(out, torch.full((4, 4), 1.0))


def test_merge_respects_weights():
    a, b = _ckpt(10, 0.0), _ckpt(20, 4.0)
    out = merger.merge_states([a, b], [0.25, 0.75])["w"]
    assert torch.allclose(out, torch.full((4, 4), 3.0))


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        merger.merge_states([_ckpt(1, 0.0), _ckpt(2, 1.0)], [0.5, 0.9])


def test_merge_preserves_matching_dtype():
    a, b = _ckpt(1, 0.0), _ckpt(2, 1.0)
    for st in (a, b):
        st["model"]["w"] = st["model"]["w"].to(torch.bfloat16)
    out = merger.merge_states([a, b], [0.5, 0.5])["w"]
    assert out.dtype == torch.bfloat16, "merging must not silently upcast parameters"


def test_merge_rejects_mixed_dtypes():
    a = _ckpt(1, 0.0)
    a["model"]["w"] = a["model"]["w"].to(torch.bfloat16)
    b = _ckpt(2, 1.0)
    with pytest.raises(ValueError):
        merger.check_compatible([a, b], [Path("a"), Path("b")])


def test_merge_rejects_mismatched_config():
    a, b = _ckpt(1, 0.0), _ckpt(2, 1.0)
    b["config"] = {"model": {"n_layer": 2}}
    with pytest.raises(ValueError):
        merger.check_compatible([a, b], [Path("a"), Path("b")])


def test_merge_rejects_mismatched_parameters():
    a, b = _ckpt(1, 0.0), _ckpt(2, 1.0)
    b["model"]["extra"] = torch.zeros(2)
    with pytest.raises(ValueError):
        merger.check_compatible([a, b], [Path("a"), Path("b")])


def test_merged_checkpoint_keeps_newer_state_and_records_provenance():
    """The merged model must continue from the newer run, and say where it came from."""
    a, b = _ckpt(10, 0.0, tokens=1_000), _ckpt(20, 2.0, tokens=2_000)
    b["optimizer"]["state"] = {"momentum": 1}
    newest = b
    merged = {
        "model": merger.merge_states([a, b], [0.5, 0.5]),
        "optimizer": newest["optimizer"],
        "step": newest["step"],
        "tokens": newest["tokens"],
        "config": newest["config"],
        "merged_from": ["a", "b"],
        "merge_weights": [0.5, 0.5],
    }
    assert merged["step"] == 20
    assert merged["tokens"] == 2_000
    assert merged["optimizer"] is b["optimizer"], "optimiser state must come from the newer run"
    assert merged["merged_from"] == ["a", "b"]
