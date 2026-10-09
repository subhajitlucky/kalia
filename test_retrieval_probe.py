"""Tests for the retrieval-utilization probe harness."""

import numpy as np
import torch

from model import GPT, GPTConfig
from retrieval_probe import (
    answer_id,
    bootstrap_ci,
    build_distraction_items,
    build_fact_items,
    mean_target_nll,
    run_best_of_n,
    run_distraction_conditions,
    run_fact_conditions,
    score_next_token,
)


class FakeEncoder:
    """One token per word; token ids stay below 128 so a 256-vocab model fits."""

    @staticmethod
    def _tok(word: str) -> int:
        return sum(ord(c) for c in word) % 128

    def encode_ordinary(self, text: str) -> list[int]:
        return [self._tok(word) for word in text.split()]


def tiny_model(vocab_size: int = 256, context_len: int = 256) -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=vocab_size, n_layer=1, n_head=2, n_embd=32, context_len=context_len))


def test_fact_items_are_deterministic_and_well_formed():
    encoder = FakeEncoder()
    first = build_fact_items(encoder, n=6, seed=7)
    second = build_fact_items(encoder, n=6, seed=7)
    assert [item["sentence"] for item in first] == [item["sentence"] for item in second]
    for item in first:
        assert item["pool"][item["true_index"]] == item["sentence"]
        assert item["answer_id"] == answer_id(encoder, item["place"])
        assert item["prompt"].endswith("hidden in the")


def test_rank_is_one_exactly_when_the_answer_is_top1():
    model = tiny_model()
    ids = [1, 2, 3, 4, 5]
    rank, logprob, top1 = score_next_token(model, ids, answer=6)
    assert rank >= 1
    assert top1 == (rank == 1)
    assert logprob <= 0.0


def test_fact_conditions_report_all_metrics_and_bm25_retrieves_the_true_passage():
    model = tiny_model()
    encoder = FakeEncoder()
    items = build_fact_items(encoder, n=6, seed=0)
    report = run_fact_conditions(model, encoder, items)
    assert report["n"] == 6
    for condition in ("none", "bm25", "oracle"):
        row = report[condition]
        assert 0.0 <= row["top1"] <= 1.0
        assert 0.0 < row["mrr"] <= 1.0
        assert row["mean_logprob"] <= 0.0
    assert report["bm25_hit_rate"] >= 0.5, "BM25 should retrieve the exact-match passage"
    assert report["bm25_true_rank_mrr"] > 0.0


def test_distraction_windows_pair_each_window_with_a_mismatched_context():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 256, size=4000).astype(np.uint16)
    windows = build_distraction_items(ids, n=6, seed=3)
    assert len(windows) == 6
    for window in windows:
        assert window["irrelevant_context_ids"] != window["context_ids"]
        assert len(window["context_ids"]) == 64
        assert len(window["target_ids"]) == 16


def test_distraction_conditions_report_finite_nlls():
    model = tiny_model()
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 256, size=4000).astype(np.uint16)
    windows = build_distraction_items(ids, n=4, seed=5)
    report = run_distraction_conditions(model, windows)
    for condition in ("bare", "relevant", "irrelevant"):
        assert np.isfinite(report[condition]["mean_nll"])
        assert report[condition]["mean_nll"] > 0.0


def test_mean_target_nll_with_empty_context_matches_manual_forward():
    model = tiny_model()
    prompt = [1, 2, 3]
    target = [4, 5]
    nll = mean_target_nll(model, [], prompt, target)
    logits, _ = model(torch.tensor([prompt + target]))
    logp = torch.log_softmax(logits[0], dim=-1)
    expected = -(logp[2, 4].item() + logp[3, 5].item()) / 2
    assert abs(nll - expected) < 1e-6


def test_bootstrap_ci_is_deterministic_and_bounds_the_mean():
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    lo, hi = bootstrap_ci(values, seed=0)
    lo2, hi2 = bootstrap_ci(values, seed=0)
    assert (lo, hi) == (lo2, hi2)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi
    constant = bootstrap_ci([2.0, 2.0, 2.0], seed=0)
    assert constant == (2.0, 2.0)


def test_best_of_n_selection_cannot_beat_coverage():
    model = tiny_model()
    encoder = FakeEncoder()
    items = build_fact_items(encoder, n=4, seed=11)
    report = run_best_of_n(model, encoder, items, samples=3, seed=0)
    assert report["n"] == 4
    assert report["selected_top1"] <= report["coverage"] + 1e-9
    assert report["score_token_windows"] == [1, 2, 4]
    for k in report["score_token_windows"]:
        assert report[f"sel_k{k}_top1"] <= report["coverage"] + 1e-9
    for key in ("greedy_top1", "selected_top1", "majority_top1", "coverage"):
        assert 0.0 <= report[key] <= 1.0
