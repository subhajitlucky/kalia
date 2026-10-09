"""Tests for entity-verified decoding.

The verifier is deterministic and rule-based (eval_entities.py); these tests
pin the selection order -- retention first, span second, log-probability last --
and that the whole loop is seed-reproducible.
"""

import torch

from entity_decode import entity_decode, score_candidate, select_best_index
from model import GPT, GPTConfig


class WordEncoder:
    """Word-level stand-in with a reversible encode/decode pair."""

    VOCAB = [
        "Max", "Lily", "the", "park", "ran", "and", "to", "was", "happy", "sad",
        "Tim", "house", "slept", "home", "Then", "he", "a", "cat", "named",
        "Once", "upon", "time", "there", "went", "again",
    ]

    def __init__(self) -> None:
        self._ids = {word: i for i, word in enumerate(self.VOCAB)}

    def encode_ordinary(self, text: str) -> list[int]:
        return [self._ids.get(word, 0) for word in text.split()]

    def decode(self, ids) -> str:
        return " ".join(self.VOCAB[int(i) % len(self.VOCAB)] for i in ids)


def test_retention_dominates_the_ranking():
    prompt = "Max and Lily went to the park."
    keeps_both = "Max and Lily ran to the park and Max was happy."
    loses_lily = "Max ran to the park and was happy."
    keeps = score_candidate(prompt, keeps_both)
    loses = score_candidate(prompt, loses_lily)
    assert keeps["retention"] > loses["retention"]
    assert select_best_index([loses, keeps]) == 1


def test_span_breaks_retention_ties():
    prompt = "Max went to the park."
    early_only = "Max went home. Then he slept. And slept. And slept."
    both_ends = "Max went home. Then he slept. And Max slept."
    a = score_candidate(prompt, early_only)
    b = score_candidate(prompt, both_ends)
    assert a["retention"] == b["retention"] == 1.0
    assert b["mean_span"] > a["mean_span"]
    assert select_best_index([a, b]) == 1


def test_logprob_breaks_full_ties():
    prompt = "Max went to the park."
    text = "Max went home."
    weak = score_candidate(prompt, text, logprob=-3.0)
    strong = score_candidate(prompt, text, logprob=-1.0)
    assert weak["retention"] == strong["retention"]
    assert weak["mean_span"] == strong["mean_span"]
    assert select_best_index([weak, strong]) == 1


def test_prompts_without_entities_default_to_full_retention():
    row = score_candidate("the cat sat", "the cat slept")
    assert row["retention"] == 1.0


def test_empty_score_list_is_refused():
    import pytest

    with pytest.raises(ValueError):
        select_best_index([])


def test_entity_decode_picks_the_argmax_and_is_seed_deterministic():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, n_layer=1, n_head=2, n_embd=32, context_len=32))
    encoder = WordEncoder()
    prompt = "Max and Lily went to the park."
    first = entity_decode(
        model, encoder, prompt, n=4, max_new_tokens=6, temperature=1.0, top_k=None, seed=7
    )
    assert len(first.texts) == 4
    assert len(first.scores) == 4
    recomputed = [
        score_candidate(prompt, text, first.scores[i]["logprob"])
        for i, text in enumerate(first.texts)
    ]
    assert first.chosen == select_best_index(recomputed)

    second = entity_decode(
        model, encoder, prompt, n=4, max_new_tokens=6, temperature=1.0, top_k=None, seed=7
    )
    assert second.texts == first.texts
    assert second.chosen == first.chosen


def test_single_candidate_cannot_be_reranked():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, n_layer=1, n_head=2, n_embd=32, context_len=32))
    result = entity_decode(
        model, WordEncoder(), "Max went to the park.", n=1, max_new_tokens=4, seed=0
    )
    assert result.chosen == 0
