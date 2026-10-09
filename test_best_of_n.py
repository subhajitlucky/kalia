"""Tests for best-of-N re-ranking with the model's own verifier."""

import pytest
import torch

from best_of_n import best_of_n, continuation_logprob, select_best
from model import GPT, GPTConfig


def tiny_model(vocab_size: int = 64, context_len: int = 32) -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=vocab_size, n_layer=1, n_head=2, n_embd=32, context_len=context_len))


def test_continuation_logprob_matches_manual_computation():
    model = tiny_model()
    seq = torch.arange(12).remainder(model.cfg.vocab_size).unsqueeze(0)
    prompt, cont = seq[:, :5], seq[:, 5:]
    total, mean = continuation_logprob(model, prompt, cont)

    logits, _ = model(seq)
    logp = torch.log_softmax(logits[0], dim=-1)
    positions = [prompt.size(1) - 1 + i for i in range(cont.size(1))]
    expected = sum(logp[p, seq[0, p + 1]].item() for p in positions)

    assert abs(total - expected) < 1e-5
    assert abs(mean - expected / cont.size(1)) < 1e-5


def test_select_best_normalised_and_raw_disagree_on_unequal_lengths():
    totals, lengths = [-6.0, -9.0], [3, 6]
    assert select_best(totals, lengths, length_normalize=True) == 1
    assert select_best(totals, lengths, length_normalize=False) == 0


def test_select_best_ties_break_to_the_earliest():
    assert select_best([-5.0, -5.0], [2, 2]) == 0


def test_select_best_validates_its_inputs():
    with pytest.raises(ValueError):
        select_best([], [])
    with pytest.raises(ValueError):
        select_best([-1.0], [1, 2])
    with pytest.raises(ValueError):
        select_best([-1.0], [0])


def test_best_of_n_selects_the_highest_rescored_candidate():
    model = tiny_model()
    prompt = torch.tensor([1, 2, 3, 4])
    result = best_of_n(model, prompt, n=4, max_new_tokens=6, temperature=1.0, top_k=None, seed=11)

    assert len(result.sequences) == 4
    rescored = [
        continuation_logprob(model, prompt, seq[prompt.numel() :])[1]
        for seq in result.sequences
    ]
    assert rescored[result.chosen] >= max(rescored) - 1e-9
    assert torch.equal(result.ids, result.sequences[result.chosen])


def test_max_tokens_scores_only_the_first_k_tokens():
    model = tiny_model()
    seq = torch.arange(10).remainder(model.cfg.vocab_size).unsqueeze(0)
    prompt, cont = seq[:, :4], seq[:, 4:]
    full_total, _ = continuation_logprob(model, prompt, cont)
    first_total, first_mean = continuation_logprob(model, prompt, cont, max_tokens=1)

    logits, _ = model(seq)
    logp = torch.log_softmax(logits[0], dim=-1)
    expected_first = logp[3, seq[0, 4]].item()

    assert abs(first_total - expected_first) < 1e-5
    assert abs(first_mean - expected_first) < 1e-5
    assert full_total <= first_total + 1e-9


def test_best_of_n_score_tokens_reproduces_first_token_selection():
    model = tiny_model()
    prompt = torch.tensor([1, 2, 3, 4])
    result = best_of_n(
        model, prompt, n=4, max_new_tokens=4, temperature=1.0, top_k=None, seed=5, score_tokens=1
    )
    rescored = [
        continuation_logprob(model, prompt, seq[prompt.numel() :], max_tokens=1)[1]
        for seq in result.sequences
    ]
    assert rescored[result.chosen] >= max(rescored) - 1e-9


def test_same_seed_reproduces_the_same_candidates_and_winner():
    model = tiny_model()
    prompt = torch.tensor([1, 2, 3])
    a = best_of_n(model, prompt, n=4, max_new_tokens=6, seed=3)
    b = best_of_n(model, prompt, n=4, max_new_tokens=6, seed=3)
    assert a.chosen == b.chosen
    assert torch.equal(a.ids, b.ids)
    for left, right in zip(a.sequences, b.sequences):
        assert left.tolist() == right.tolist()


def test_single_candidate_cannot_be_reranked():
    model = tiny_model()
    result = best_of_n(model, torch.tensor([1, 2]), n=1, max_new_tokens=3, seed=0)
    assert result.chosen == 0
    assert result.ids.numel() == 5


def test_prompts_and_n_are_validated():
    model = tiny_model()
    with pytest.raises(ValueError):
        best_of_n(model, torch.tensor([1, 2]), n=0)
    with pytest.raises(ValueError):
        best_of_n(model, torch.tensor([], dtype=torch.long), n=2)
    with pytest.raises(ValueError):
        continuation_logprob(model, torch.tensor([], dtype=torch.long), torch.tensor([1]))
