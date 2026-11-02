"""Tests for the clean 2x2 composition benchmark and its scorer."""

from __future__ import annotations

import json

import torch

from composition_benchmark import (
    CELL_ID,
    NAMES_BENCH,
    audit,
    generate_benchmark,
    write_benchmark,
)
from eval_ip import accuracy_from_scores, score_candidates
from intervention_pairs import NAMES_EVAL, NAMES_TRAIN, InterventionDataset
from model import GPT, GPTConfig


def test_deterministic_and_four_cells():
    a = generate_benchmark(seed=7, n_per_cell=20)
    b = generate_benchmark(seed=7, n_per_cell=20)
    assert a == b
    assert len(a) == 80
    cells = {r["cell"] for r in a}
    assert cells == set(CELL_ID.values())
    for cell in CELL_ID.values():
        assert sum(1 for r in a if r["cell"] == cell) == 20


def test_cells_separate_template_and_composition():
    records = generate_benchmark(seed=7, n_per_cell=20)
    for r in records:
        if r["cell"] in ("S-S", "S-P"):
            assert r["template_familiarity"] == "seen"
        else:
            assert r["template_familiarity"] == "unseen"
        if r["cell"] in ("S-S", "U-S"):
            assert r["composition_type"] == "single"
            assert r["n_way"] == 2
            assert not r["composition"]
        else:
            assert r["composition_type"] == "pair"
            assert r["n_way"] == 4
            assert r["composition"]


def test_names_are_new_and_shared():
    records = generate_benchmark(seed=7, n_per_cell=20)
    names = {v for r in records for v in r["entities"].values() if v}
    assert names <= set(NAMES_BENCH)
    assert not names & set(NAMES_TRAIN)
    assert not names & set(NAMES_EVAL)
    assert names


def test_candidates_are_distinct_and_correct():
    records = generate_benchmark(seed=7, n_per_cell=20)
    for r in records:
        assert len(set(r["candidates"])) == len(r["candidates"])
        assert r["candidates"][r["correct_index"]] == r["suffix"]
        assert r["suffix"] != r["factual_suffix"]
        assert r["edited_prefix"] != r["prefix"]
        assert r["n_way"] == len(r["candidates"])


def test_audit_and_write(tmp_path):
    manifest = write_benchmark(tmp_path, seed=7, n_per_cell=20)
    assert manifest["n"] == 80
    assert set(manifest["cells"]) == set(CELL_ID.values())
    assert all(cell["n"] == 20 for cell in manifest["cells"].values())
    assert len(manifest["sha256"]) == 64
    loaded = json.loads((tmp_path / "audit.json").read_text())
    assert loaded["n"] == 80
    assert set(loaded["examples"]) == set(CELL_ID.values())
    # Audit is a pure check; it must accept the generated set.
    audit(generate_benchmark(seed=7, n_per_cell=20))


def test_perfect_scores_fill_every_cell(tmp_path):
    write_benchmark(tmp_path, seed=7, n_per_cell=20)
    ds = InterventionDataset(tmp_path / "bench.jsonl", prefix_len=64, suffix_len=64)
    items = [
        (i, j) for i, r in enumerate(ds.records) for j in range(len(r["candidates"]))
    ]
    perfect = {
        item: (0.0 if item[1] == ds.records[item[0]]["correct_index"] else 1.0)
        for item in items
    }
    acc = accuracy_from_scores(ds, perfect)
    assert set(acc["by_cell"]) == set(CELL_ID.values())
    for cell, bucket in acc["by_cell"].items():
        assert bucket["n"] == 20
        assert bucket["accuracy"] == 1.0
    assert acc["seen_n"] == 40
    assert acc["unseen_n"] == 40
    assert acc["by_cell"]["S-S"]["chance"] == 0.5
    assert acc["by_cell"]["S-P"]["chance"] == 0.25


def test_score_candidates_on_a_tiny_model(tmp_path):
    write_benchmark(tmp_path, seed=7, n_per_cell=4)
    ds = InterventionDataset(tmp_path / "bench.jsonl", prefix_len=64, suffix_len=64)
    model = GPT(GPTConfig(
        vocab_size=50257, n_layer=1, n_head=2, n_embd=32,
        context_len=128, dropout=0.0, qk_norm=True,
    )).eval()
    scores = score_candidates(model, ds, torch.device("cpu"), batch_size=8)
    expected = sum(len(r["candidates"]) for r in ds.records)
    assert len(scores) == expected
    acc = accuracy_from_scores(ds, scores)
    assert set(acc["by_cell"]) == set(CELL_ID.values())
    assert sum(b["n"] for b in acc["by_cell"].values()) == len(ds.records)
