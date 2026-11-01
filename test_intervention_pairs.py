"""X40: tests for the intervention pair generator and dataset.

The generator is the evidential base for I-3 (unseen compositions). These tests
pin the things that would make that bar meaningless:

1. **Leakage** — train and eval must not share templates or names, and train
   must contain singles only.
2. **Well-posedness** — the model must see the factual prefix plus an explicit
   intervention instruction, not the already-edited prefix. The edited prefix
   is only the next-state target.
3. **The composition set** — four unseen compositions, each with a composed
   correct answer and three single/factual distractors.
4. **Masks** — the suffix loss and the workspace write must exclude padding,
   and the target must be the edited prefix, never the factual one.
"""

from __future__ import annotations

import json
import re

import pytest
import torch

from intervention_pairs import (
    COMPOSITIONS,
    OP_DIM,
    OP_NAMES,
    PAD_ID,
    InterventionDataset,
    audit,
    causal_pad_mask,
    generate,
    render,
    write_pairs,
)


def test_deterministic():
    assert generate(seed=7, n_train=20, n_eval=10, n_comp=10) == generate(
        seed=7, n_train=20, n_eval=10, n_comp=10
    )


def test_train_eval_disjoint_and_clean():
    train, eval_ = generate(seed=7, n_train=20, n_eval=10, n_comp=10)
    manifest = audit(train, eval_)
    assert not set(manifest["train_templates"]) & set(manifest["eval_templates"])
    assert not set(manifest["train_names"]) & set(manifest["eval_names"])
    assert all(not row["composition"] for row in train)
    assert all(row["n_way"] == 1 for row in train)
    assert all(row["n_way"] == len(row["candidates"]) for row in eval_)
    assert all(len(set(row["candidates"])) == len(row["candidates"]) for row in eval_)


def test_all_train_ops_and_all_compositions_present():
    train, eval_ = generate(seed=7, n_train=40, n_eval=10, n_comp=10)
    assert {row["operations"][0] for row in train} == set(OP_NAMES)
    comps = {tuple(row["operations"]) for row in eval_ if row["composition"]}
    assert comps == set(COMPOSITIONS)
    assert all(row["n_way"] == 4 for row in eval_ if row["composition"])
    assert all(row["n_way"] == 2 for row in eval_ if not row["composition"])


def test_model_sees_factual_prefix_and_instruction():
    train, eval_ = generate(seed=7, n_train=20, n_eval=10, n_comp=10)
    for row in train + eval_:
        assert row["prefix"] != row["edited_prefix"]
        assert row["suffix"] != row["factual_suffix"]
        assert row["instruction"].startswith("Intervention: ")
        assert row["instruction"].endswith(".")
        # The input prefix is the factual chain: it must not contain the
        # post-intervention suffix.
        assert row["suffix"] not in row["prefix"]
        # The edited prefix is only for the target, never the prompt.
        assert row["edited_prefix"] != row["prefix"]


def test_rename_instruction_names_the_mapping():
    train, _ = generate(seed=7, n_train=30, n_eval=10, n_comp=10)
    rows = [row for row in train if row["operations"] == ["rename"]]
    assert rows
    for row in rows:
        e = row["entities"]
        pairs = [(e["A"], e["A2"]), (e["B"], e["B2"])]
        if e["C"]:
            pairs.append((e["C"], e["C2"]))
        factual = f"{row['prefix']} {row['factual_suffix']}"
        edited = f"{row['edited_prefix']} {row['suffix']}"
        for old, new in pairs:
            assert re.search(rf"\b{old}\b", factual)
            assert re.search(rf"\b{new}\b", edited)
            assert f"{old} to {new}" in row["instruction"]


def test_swap_swaps_the_two_participants():
    train, _ = generate(seed=7, n_train=30, n_eval=10, n_comp=10)
    rows = [row for row in train if row["operations"] == ["swap"]]
    assert rows
    for row in rows:
        e = row["entities"]
        # The factual prefix starts with A, the edited prefix with B.
        assert row["prefix"].startswith(e["A"] + " ")
        assert row["edited_prefix"].startswith(e["B"] + " ")
        assert f"swap {e['A']} and {e['B']}" in row["instruction"]


def test_negate_changes_the_event():
    train, _ = generate(seed=7, n_train=30, n_eval=10, n_comp=10)
    rows = [row for row in train if row["operations"] == ["negate"]]
    assert rows
    for row in rows:
        assert "not" in row["edited_prefix"]
        assert "not" not in row["prefix"]
        assert "negate the event" in row["instruction"]


def test_number_and_place_change_the_target():
    train, _ = generate(seed=7, n_train=40, n_eval=10, n_comp=10)
    numbers = [row for row in train if row["operations"] == ["number"]]
    places = [row for row in train if row["operations"] == ["place"]]
    assert numbers and places
    for row in numbers:
        assert "two" in row["prefix"]
        assert "three" in row["suffix"]
        assert "change two to three" in row["instruction"]
    for row in places:
        assert " on " in row["prefix"]
        assert " under " in row["suffix"]
        assert "change on to under" in row["instruction"]


def test_composition_candidates_include_single_op_distractors():
    _, eval_ = generate(seed=7, n_train=20, n_eval=10, n_comp=10)
    rows = [row for row in eval_ if row["composition"]]
    assert rows
    for row in rows:
        correct = row["candidates"][row["correct_index"]]
        assert correct == row["suffix"]
        assert row["factual_suffix"] in row["candidates"]
        assert len(row["candidates"]) == 4
        # Every alternative is a real alternative, not the correct answer.
        assert all(c != correct for i, c in enumerate(row["candidates"])
                   if i != row["correct_index"])


def test_audit_rejects_leakage():
    train, eval_ = generate(seed=7, n_train=20, n_eval=10, n_comp=10)
    with pytest.raises(AssertionError):
        audit(train, train)
    with pytest.raises(AssertionError):
        audit(train + [dict(train[0], composition=True)], eval_)


def test_write_pairs_manifest(tmp_path):
    manifest = write_pairs(tmp_path, seed=7, n_train=20, n_eval=10, n_comp=10)
    assert manifest["n_train"] == 80
    assert manifest["n_eval"] == 70
    assert len(manifest["train_sha256"]) == 64
    assert len(manifest["eval_sha256"]) == 64
    assert manifest["lengths"]["train_prompt_tokens"] <= 64
    assert manifest["lengths"]["eval_prompt_tokens"] <= 64
    loaded = json.loads((tmp_path / "audit.json").read_text())
    assert len(loaded["examples"]) == len(set(OP_NAMES)) + len(COMPOSITIONS)


def test_dataset_batch_masks_and_target(tmp_path):
    write_pairs(tmp_path, seed=7, n_train=20, n_eval=10, n_comp=10)
    ds = InterventionDataset(tmp_path / "train.jsonl")
    generator = torch.Generator().manual_seed(0)
    batch = ds.get_batch(8, torch.device("cpu"), generator)
    assert batch.x.shape == (8, 128)
    assert batch.y.shape == (8, 128)
    assert batch.attn_mask.shape == (8, 1, 128, 128)
    assert batch.op_vec.shape == (8, OP_DIM)
    assert batch.target_x.shape == (8, 64)
    assert batch.target_mask.shape == (8, 64)
    assert bool((batch.loss_mask.sum(dim=1) > 0).all())
    assert bool((batch.token_mask.sum(dim=1) > 0).all())
    assert bool((batch.target_mask.sum(dim=1) > 0).all())
    # The prompt is left-padded, so the first suffix token sits at index 64.
    assert bool((batch.loss_mask[:, 63]).all())
    # Nothing outside the real suffix is supervised.
    assert not bool(batch.loss_mask[:, :63].any())
    assert not bool(batch.loss_mask[:, 64 + batch.loss_mask[:, 64:].sum(dim=1).max():].any())


def test_dataset_target_is_the_edited_prefix(tmp_path):
    write_pairs(tmp_path, seed=7, n_train=20, n_eval=10, n_comp=10)
    ds = InterventionDataset(tmp_path / "train.jsonl")
    for i in range(min(10, len(ds))):
        row = ds.records[i]
        _, _, _, _, target, target_mask, _ = ds.example(i)
        ids = [int(t) for t in target[target_mask]]
        text = ds.enc.decode(ids)
        assert text == row["edited_prefix"]


def test_candidate_encoding(tmp_path):
    write_pairs(tmp_path, seed=7, n_train=20, n_eval=10, n_comp=10)
    ds = InterventionDataset(tmp_path / "eval.jsonl")
    row = ds.records[0]
    for candidate in row["candidates"]:
        x, y, loss_mask, token_mask = ds.candidate(0, candidate)
        assert x.shape == (128,)
        assert y.shape == (128,)
        assert bool(loss_mask.any())
        assert bool(token_mask[:64].any())
        assert bool(token_mask[64:].any())
        assert int(loss_mask.sum()) == len(
            ds.enc.encode_ordinary(candidate)
        )


def test_causal_pad_mask_blocks_padding():
    token_mask = torch.tensor([[False, False, True, True]])
    mask = causal_pad_mask(token_mask)
    assert mask.shape == (1, 1, 4, 4)
    # A real position cannot attend to a padded key.
    assert not bool(mask[0, 0, 2, 0])
    assert not bool(mask[0, 0, 2, 1])
    assert bool(mask[0, 0, 2, 2])
    # Causality: position 2 cannot see position 3.
    assert not bool(mask[0, 0, 2, 3])


def test_render_is_order_sensitive_for_compositions():
    from intervention_pairs import TRAIN_TEMPLATES, _slots
    import random

    rng = random.Random(0)
    template = next(t for t in TRAIN_TEMPLATES if t["id"] == "give_keep")
    slots = _slots(rng, ["Maya", "Leo", "Nora", "Sam"], template)
    rename_then_swap = render(template, slots, ["rename", "swap"])
    swap_then_rename = render(template, slots, ["swap", "rename"])
    assert rename_then_swap[1] != swap_then_rename[1]
    assert rename_then_swap[2] != swap_then_rename[2]
