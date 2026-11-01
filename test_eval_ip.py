"""X40: tests for the seen/unseen scorer and workspace diagnostics."""

from __future__ import annotations

import math

import torch

from eval_ip import accuracy_from_scores, score_candidates, workspace_stats
from intervention_pairs import InterventionDataset, write_pairs
from model import GPT, GPTConfig

TINY = dict(
    vocab_size=50257, n_layer=2, n_head=2, n_embd=32, context_len=64,
    dropout=0.0, qk_norm=True, logit_softcap=30.0,
)
EVENT = dict(
    event_slots=4, event_slot_dim=8, event_chunk=48, event_head=True,
)


def make_ds(tmp_path):
    write_pairs(tmp_path, seed=7, n_train=10, n_eval=5, n_comp=5)
    return InterventionDataset(tmp_path / "eval.jsonl", prefix_len=48, suffix_len=16)


def test_score_candidates_covers_every_candidate(tmp_path):
    ds = make_ds(tmp_path)
    torch.manual_seed(0)
    model = GPT(GPTConfig(**TINY, **EVENT)).eval()
    scores = score_candidates(model, ds, torch.device("cpu"), batch_size=8)
    expected = sum(len(r["candidates"]) for r in ds.records)
    assert len(scores) == expected
    assert all(math.isfinite(v) for v in scores.values())
    assert all(v > 0 for v in scores.values())


def test_accuracy_uses_the_lowest_nll(tmp_path):
    ds = make_ds(tmp_path)
    items = [
        (i, j) for i, r in enumerate(ds.records) for j in range(len(r["candidates"]))
    ]
    perfect = {
        item: (0.0 if item[1] == ds.records[item[0]]["correct_index"] else 1.0)
        for item in items
    }
    result = accuracy_from_scores(ds, perfect)
    assert result["seen"] == 1.0
    assert result["unseen"] == 1.0
    assert result["overall"] == 1.0
    assert result["seen_n"] + result["unseen_n"] == len(ds.records)
    assert result["unseen_n"] == 20

    inverted = {item: 1.0 - value for item, value in perfect.items()}
    result = accuracy_from_scores(ds, inverted)
    assert result["overall"] == 0.0
    assert result["seen_chance"] == 0.5
    assert result["unseen_chance"] == 0.25


def test_accuracy_by_operation_is_reported(tmp_path):
    ds = make_ds(tmp_path)
    scores = {
        (i, j): 0.5
        for i, r in enumerate(ds.records)
        for j in range(len(r["candidates"]))
    }
    result = accuracy_from_scores(ds, scores)
    assert result["by_operation"]
    assert all("accuracy" in bucket for bucket in result["by_operation"].values())
    assert sum(b["n"] for b in result["by_operation"].values()) == len(ds.records)


def test_workspace_stats_are_finite(tmp_path):
    ds = make_ds(tmp_path)
    torch.manual_seed(0)
    model = GPT(GPTConfig(**TINY, **EVENT)).eval()
    stats = workspace_stats(model, ds, torch.device("cpu"), batch_size=8)
    assert stats["n"] == len(ds.records)
    assert stats["slots"] == 4
    assert stats["slot_dim"] == 8
    assert stats["slot_entropy_bits"] >= 0
    assert stats["effective_rank"] > 0
    assert 0 <= stats["probe_accuracy"] <= 1
    assert -1 <= stats["probe_gain"] <= 1
    assert stats["probe_classes"]


def test_workspace_stats_empty_without_a_workspace(tmp_path):
    ds = make_ds(tmp_path)
    model = GPT(GPTConfig(**TINY))
    assert workspace_stats(model, ds, torch.device("cpu")) == {}
