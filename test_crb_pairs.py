"""X38: tests for the counterfactual pair generator and dataset."""

import json
import re

import pytest
import torch

from crb_pairs import CounterfactualPairDataset, audit, generate, write_pairs


def test_deterministic():
    a = generate(seed=7, n_train=20, n_eval=10)
    b = generate(seed=7, n_train=20, n_eval=10)
    assert a == b


def test_train_eval_disjoint():
    train, eval_ = generate(seed=7, n_train=20, n_eval=10)
    manifest = audit(train, eval_)
    assert not set(manifest["train_templates"]) & set(manifest["eval_templates"])
    assert not set(manifest["train_names"]) & set(manifest["eval_names"])


def test_all_subtypes_present():
    train, eval_ = generate(seed=7, n_train=30, n_eval=12)
    subtypes = {row["subtype"] for row in train + eval_}
    assert subtypes == {"rename", "synonym", "swap", "negate", "number", "place"}


def test_rename_is_consistent():
    train, _ = generate(seed=7, n_train=20, n_eval=10)
    rows = [row for row in train if row["subtype"] == "rename"]
    assert rows
    for row in rows:
        detail = row["edit_detail"]
        for old, new in ((detail["A"], detail["A2"]), (detail["B"], detail["B2"])):
            if re.search(rf"\b{re.escape(old)}\b", row["original"]):
                assert not re.search(rf"\b{re.escape(old)}\b", row["edited"])
                assert re.search(rf"\b{re.escape(new)}\b", row["edited"])


def test_synonym_changes_the_adjective():
    train, _ = generate(seed=7, n_train=20, n_eval=10)
    rows = [row for row in train if row["subtype"] == "synonym"]
    assert rows
    for row in rows:
        detail = row["edit_detail"]
        assert detail["adj"] in row["original"]
        assert detail["synonym"] in row["edited"]


def test_negation_adds_not():
    train, _ = generate(seed=7, n_train=20, n_eval=10)
    rows = [row for row in train if row["subtype"] == "negate"]
    assert rows
    for row in rows:
        assert "not" in row["edited"]
        assert "not" not in row["original"]


def test_number_changes():
    train, _ = generate(seed=7, n_train=20, n_eval=10)
    rows = [row for row in train if row["subtype"] == "number"]
    assert rows
    for row in rows:
        assert "two" in row["original"]
        assert "three" in row["edited"]


def test_place_changes_preposition():
    train, eval_ = generate(seed=7, n_train=20, n_eval=10)
    rows = [row for row in train + eval_ if row["subtype"] == "place"]
    assert rows
    for row in rows:
        left, right = row["edit_detail"]["prep"].split(" -> ")
        assert left in row["original"]
        assert right in row["edited"]


def test_write_pairs_manifest(tmp_path):
    manifest = write_pairs(tmp_path, seed=7, n_train=20, n_eval=10)
    assert manifest["n_train"] == 120
    assert manifest["n_eval"] == 20
    assert len(manifest["train_sha256"]) == 64
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "eval.jsonl").exists()
    assert (tmp_path / "audit.json").exists()
    loaded = json.loads((tmp_path / "audit.json").read_text())
    assert len(loaded["examples"]) == 6


def test_dataset_batch(tmp_path):
    write_pairs(tmp_path, seed=7, n_train=20, n_eval=10)
    ds = CounterfactualPairDataset(tmp_path / "train.jsonl", max_len=32)
    generator = torch.Generator().manual_seed(0)
    batch = ds.get_batch(8, torch.device("cpu"), generator)
    assert batch.orig.shape == (8, 32)
    assert batch.edit.shape == (8, 32)
    assert batch.types.shape == (8,)
    assert set(batch.types.tolist()) == {0, 1}
    assert bool((batch.orig_mask.sum(dim=1) > 0).all())
    assert bool((batch.edit_mask.sum(dim=1) > 0).all())


def test_dataset_requires_both_types(tmp_path):
    rows = [row for row in generate(seed=7, n_train=20, n_eval=10)[0]
            if row["edit_type"] == "invariant"]
    path = tmp_path / "invariant_only.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError):
        CounterfactualPairDataset(path, max_len=32)
