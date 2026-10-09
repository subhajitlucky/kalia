"""Tests for the Kautilya mixture loader.

The v0.3.0 pre-registration lists runtime per-source weighting as the largest
unwritten piece of 0.3.0, and says Step 4 defers to 0.4 if it is not ready and
tested. These tests are that condition. A reweighting policy that silently samples
the wrong proportion, or one whose "improving" arm is driven by training loss
instead of held-out loss, would produce a plausible number that means nothing --
which is the D48 failure mode in a new costume.

Everything here is CPU-only and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mixture import (
    BHEDA,
    DANA,
    DANDA,
    SAMA,
    SourceMixtureDataset,
    per_source_loss,
    strategy_counts,
)

from data import TokenDataset

CTX = 64
TOKENS = 20_000


def _ds(tmp_path, name, seed, vocab=512):
    """A small deterministic token shard."""
    p = tmp_path / f"{name}.bin"
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (TOKENS,), generator=g).to(torch.uint16)
    ids.numpy().tofile(p)
    return TokenDataset(p, context_len=CTX)


@pytest.fixture
def four_sources(tmp_path):
    return {
        "fineweb": _ds(tmp_path, "fw", 1),
        "tinystories": _ds(tmp_path, "ts", 2),
        "cosmopedia": _ds(tmp_path, "co", 3),
        "python": _ds(tmp_path, "py", 4),
    }


STATIC = {"fineweb": 0.60, "tinystories": 0.20, "cosmopedia": 0.15, "python": 0.05}


# --- construction ---------------------------------------------------------

def test_accepts_the_static_v020_mixture(four_sources):
    ds = SourceMixtureDataset(sources=four_sources, weights=STATIC)
    p = ds.probabilities()
    assert set(p) == set(four_sources)
    assert pytest.approx(sum(p.values()), abs=1e-6) == 1.0
    assert p["fineweb"] == pytest.approx(0.60, abs=0.01)


def test_weights_must_match_sources_exactly(four_sources):
    with pytest.raises(ValueError, match="do not match"):
        SourceMixtureDataset(four_sources, {k: 0.25 for k in list(four_sources)[:3]})


def test_rejects_unequal_shard_lengths(tmp_path):
    a, b = _ds(tmp_path, "a", 1), _ds(tmp_path, "b", 2)
    b.tokens = np.memmap(tmp_path / "b.bin", dtype=np.uint16, mode="r")[: TOKENS // 2]
    with pytest.raises(ValueError, match="different lengths"):
        SourceMixtureDataset({"a": a, "b": b}, {"a": 0.5, "b": 0.5})


def test_rejects_all_zero_weights(four_sources):
    with pytest.raises(ValueError, match="sum to zero"):
        SourceMixtureDataset(four_sources, {k: 0.0 for k in four_sources})


def test_context_len_matches_token_dataset(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    assert ds.context_len == CTX
    assert len(ds) == TOKENS


# --- sampling -------------------------------------------------------------

def test_realised_source_proportions_track_requested_weights(four_sources):
    """The whole point: sampling must realise the requested mixture."""
    ds = SourceMixtureDataset(four_sources, STATIC)
    g = torch.Generator().manual_seed(7)
    counts = {n: 0 for n in ds.source_names}
    for _ in range(40):
        _, _, src = ds.get_batch(256, torch.device("cpu"), generator=g, return_source=True)
        for s in src.tolist():
            counts[ds.source_names[s]] += 1
    total = sum(counts.values())
    for name, want in STATIC.items():
        got = counts[name] / total
        assert abs(got - want) < 0.03, f"{name}: sampled {got:.3f}, wanted {want:.3f}"


def test_sampling_is_deterministic_for_a_seed(four_sources):
    a = SourceMixtureDataset(four_sources, STATIC)
    b = SourceMixtureDataset(four_sources, STATIC)
    ga, gb = torch.Generator().manual_seed(11), torch.Generator().manual_seed(11)
    for _ in range(5):
        xa, ya, sa = a.get_batch(32, torch.device("cpu"), generator=ga, return_source=True)
        xb, yb, sb = b.get_batch(32, torch.device("cpu"), generator=gb, return_source=True)
        assert torch.equal(xa, xb) and torch.equal(ya, yb) and torch.equal(sa, sb)


def test_different_seeds_give_different_samples(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)
    x1, _, _ = ds.get_batch(64, torch.device("cpu"), generator=g1, return_source=True)
    x2, _, _ = ds.get_batch(64, torch.device("cpu"), generator=g2, return_source=True)
    assert not torch.equal(x1, x2)


def test_masks_and_sources_return_together(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    out = ds.get_batch(8, torch.device("cpu"), torch.Generator().manual_seed(3),
                       return_masks=True, return_source=True)
    assert len(out) == 5
    x, y, attn, loss_m, src = out
    assert x.shape == y.shape == (8, CTX)
    assert attn.shape == (8, 1, CTX, CTX)  # causal, broadcast over heads
    assert loss_m.shape == (8, CTX)
    assert src.shape == (8,)


def test_min_weight_floors_a_source_without_silencing_it(four_sources):
    w = dict(STATIC, python=0.0)
    ds = SourceMixtureDataset(four_sources, w, min_weight=0.01)
    assert ds.probabilities()["python"] >= 0.009


# --- the Kautilya policy --------------------------------------------------

def test_dana_upsamples_a_source_below_the_mean_loss(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    before = ds.probabilities()["python"]
    applied = ds.update_weights_from_signal(
        {"fineweb": 2.0, "tinystories": 2.1, "cosmopedia": 2.2, "python": 5.0}
    )
    assert applied["python"] == BHEDA
    assert ds.probabilities()["python"] < before


def test_the_lowest_loss_source_gets_dana(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    before = ds.probabilities()["fineweb"]
    applied = ds.update_weights_from_signal(
        {"fineweb": 1.0, "tinystories": 3.0, "cosmopedia": 3.1, "python": 3.2}
    )
    assert applied["fineweb"] == DANA
    assert ds.probabilities()["fineweb"] > before


def test_a_source_near_the_mean_is_left_alone(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    before = ds.probabilities()
    applied = ds.update_weights_from_signal(
        {"fineweb": 3.0, "tinystories": 3.0, "cosmopedia": 3.0, "python": 3.0}
    )
    assert set(applied.values()) == {SAMA}
    after = ds.probabilities()
    for k in before:
        assert after[k] == pytest.approx(before[k], abs=1e-6)


def test_weights_stay_normalised_across_many_updates(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    for i in range(200):
        loss = {n: 3.0 + (i % 7) * 0.3 for n in ds.source_names}
        loss["python"] = 9.0
        ds.update_weights_from_signal(loss)
        p = ds.probabilities()
        assert pytest.approx(sum(p.values()), abs=1e-5) == 1.0
        assert all(v > 0 for v in p.values())


def test_repeated_updates_do_not_explode_or_collapse(four_sources):
    """A policy that runs 500 times must not drive any source to 0 or 1."""
    ds = SourceMixtureDataset(four_sources, STATIC)
    for _ in range(500):
        ds.update_weights_from_signal(
            {"fineweb": 2.0, "tinystories": 4.0, "cosmopedia": 4.0, "python": 6.0}
        )
    p = ds.probabilities()
    assert min(p.values()) > 0.005, p
    assert max(p.values()) < 0.95, p


def test_rejects_loss_for_an_unknown_source(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    with pytest.raises(ValueError, match="unknown sources"):
        ds.update_weights_from_signal({"fineweb": 2.0, "mystery": 2.0})


def test_trend_window_upsamples_an_improving_source(four_sources):
    """A1's rule: dana goes to the source whose loss is *falling*, even when its
    level is not the lowest. The X26 pilot showed the level rule drifting into
    'eating dessert' -- feeding what is already easy."""
    ds = SourceMixtureDataset(four_sources, STATIC)
    flat = {"tinystories": 2.0, "cosmopedia": 2.0, "python": 2.0}
    for fineweb in (3.0, 3.0, 3.0):
        ds.update_weights_from_signal({"fineweb": fineweb, **flat}, trend_window=2)
    applied = ds.update_weights_from_signal({"fineweb": 2.7, **flat}, trend_window=2)
    assert applied["fineweb"] == DANA
    assert applied["python"] == SAMA


def test_trend_window_downsamples_a_worsening_source(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    flat = {"tinystories": 2.0, "cosmopedia": 2.0, "python": 2.0}
    for fineweb in (2.0, 2.0, 2.0):
        ds.update_weights_from_signal({"fineweb": fineweb, **flat}, trend_window=2)
    applied = ds.update_weights_from_signal({"fineweb": 2.4, **flat}, trend_window=2)
    assert applied["fineweb"] == BHEDA


def test_trend_window_freezes_a_collapsed_source(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    flat = {"tinystories": 2.0, "cosmopedia": 2.0, "python": 2.0}
    for fineweb in (2.0, 2.0, 2.0):
        ds.update_weights_from_signal({"fineweb": fineweb, **flat}, trend_window=2)
    applied = ds.update_weights_from_signal({"fineweb": 3.2, **flat}, trend_window=2)
    assert applied["fineweb"] == DANDA
    assert strategy_counts(applied)[DANDA] == 1


def test_trend_policy_falls_back_until_two_windows_exist(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    applied = ds.update_weights_from_signal(
        {"fineweb": 3.0, "tinystories": 2.0, "cosmopedia": 2.0, "python": 2.0},
        trend_window=3,
    )
    # A trend needs two windows; before that the level rule runs, and under it
    # the above-mean source is bheda'd.
    assert applied["fineweb"] == BHEDA


def test_strategy_counts_covers_all_four(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    applied = ds.update_weights_from_signal(
        {"fineweb": 1.0, "tinystories": 3.0, "cosmopedia": 3.05, "python": 9.0}
    )
    c = strategy_counts(applied)
    assert set(c) == {SAMA, DANA, BHEDA, DANDA}
    assert sum(c.values()) == 4


# --- per-source loss ------------------------------------------------------

class _ToyModel(torch.nn.Module):
    """Matches KALIA's GPT.forward contract: (x, targets, attn_mask, loss_mask)
    -> (logits, loss). The first version returned bare logits from a different
    signature, which passed against itself and failed the moment it met the real
    model in train.py.
    """

    def __init__(self, vocab=512):
        super().__init__()
        self.vocab = vocab
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, targets=None, attn_mask=None, loss_mask=None):
        b, t = x.shape
        return torch.zeros(b, t, self.vocab) + self.p, None


def test_per_source_loss_reports_every_source(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    loss = per_source_loss(_ToyModel(), ds, torch.device("cpu"), batches=1, batch_size=4)
    assert set(loss) == set(ds.source_names)
    assert all(v > 0 for v in loss.values())


def test_per_source_loss_is_reproducible(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    m = _ToyModel()
    a = per_source_loss(m, ds, torch.device("cpu"),
                        torch.Generator().manual_seed(5), batches=2, batch_size=4)
    b = per_source_loss(m, ds, torch.device("cpu"),
                        torch.Generator().manual_seed(5), batches=2, batch_size=4)
    assert a == b


def test_set_probabilities_stores_cpu_and_feeds_the_sampler(four_sources):
    """The DDP reweight hands over a CUDA tensor; the sampler needs CPU.

    X26: rank 1 stored the broadcast tensor as-is and the run deadlocked in
    the following step's collectives. This pins the storage contract.
    """
    ds = SourceMixtureDataset(four_sources, STATIC)
    ds.set_probabilities([0.25, 0.25, 0.25, 0.25])
    assert ds._probs.device.type == "cpu"
    assert abs(sum(ds.probabilities().values()) - 1.0) < 1e-6
    x, y = ds.get_batch(2, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert x.shape[0] == 2
    with pytest.raises(ValueError):
        ds.set_probabilities([0.5])
    with pytest.raises(ValueError):
        ds.set_probabilities([0.0, 0.0, 0.0, 0.0])


def test_per_source_loss_micro_batching_matches_the_full_batch(four_sources):
    """The X26 OOM fix must not change the number.

    The first real Kautilya run died with CUDA OOM inside this function at
    batch 16 x context 1024 (16,384 x 50,257 fp32 logits ~ 3 GiB) while the
    training process held the card. Chunking the forward is a memory fix, not
    a protocol change: the mean over the same tokens must be identical.
    """
    ds = SourceMixtureDataset(four_sources, STATIC)
    m = _ToyModel()
    full = per_source_loss(
        m, ds, torch.device("cpu"), torch.Generator().manual_seed(7), batches=2, batch_size=8
    )
    chunked = per_source_loss(
        m, ds, torch.device("cpu"), torch.Generator().manual_seed(7),
        batches=2, batch_size=8, micro_batch=3,
    )
    assert full.keys() == chunked.keys()
    for name in full:
        # Not exact equality: summing per-chunk partials and summing the full
        # batch differ in float reduction order (measured 2.4e-7 here). That is
        # arithmetic, not a change in the measurement.
        assert full[name] == pytest.approx(chunked[name], rel=1e-6, abs=1e-9), (
            f"{name}: chunking changed the number beyond float-summation order "
            f"({full[name]} vs {chunked[name]})"
        )


def test_per_source_loss_restores_training_mode(four_sources):
    ds = SourceMixtureDataset(four_sources, STATIC)
    m = _ToyModel().train()
    per_source_loss(m, ds, torch.device("cpu"), batches=1, batch_size=4)
    assert m.training is True


def test_per_source_loss_distinguishes_sources(four_sources):
    """A model that is good on one shard must show lower loss there.

    Without this, "per-source loss" could be a constant and the whole Kautilya
    signal would be noise.
    """
    ds = SourceMixtureDataset(four_sources, STATIC)

    class Skewed(torch.nn.Module):
        def __init__(self, vocab=512):
            super().__init__()
            self.vocab = vocab
            self.bias = torch.nn.Parameter(torch.zeros(vocab))

        def forward(self, x, targets=None, attn_mask=None, loss_mask=None):
            b, t = x.shape
            # Strongly favour the low token ids, which dominate one shard only.
            prior = torch.zeros(self.vocab)
            prior[:64] = 8.0
            return prior.expand(b, t, self.vocab).clone(), None

    loss = per_source_loss(Skewed(), ds, torch.device("cpu"), batches=3, batch_size=8)
    assert len(set(round(v, 6) for v in loss.values())) > 1, loss


def test_per_source_loss_actually_varies_by_source(four_sources):
    """A constant per-source loss would silently drive the policy on a fake signal.

    Found by falsification: replacing the measured value with a constant 3.0 left
    every test green, because the toy model in the other tests was uniform enough
    that a constant was indistinguishable from the truth. The policy would then
    label every source `sama` forever and the arm would report "the mixture is
    stable" when nothing had been measured at all.

    So this asserts on a model that genuinely separates the shards: the resulting
    losses must not all be equal.
    """
    ds = SourceMixtureDataset(four_sources, STATIC)

    class SharpPrior(torch.nn.Module):
        """Favours a narrow id range that one shard over-represents."""

        def __init__(self, vocab=512):
            super().__init__()
            self.vocab = vocab

        def forward(self, x, targets=None, attn_mask=None, loss_mask=None):
            b, t = x.shape
            prior = torch.zeros(self.vocab)
            prior[:8] = 12.0
            return prior.expand(b, t, self.vocab).clone(), None

    loss = per_source_loss(SharpPrior(), ds, torch.device("cpu"), batches=4, batch_size=16)
    spread = max(loss.values()) - min(loss.values())
    assert spread > 1e-6, (
        f"per-source loss is constant across sources ({spread:.2e}); the Kautilya "
        "policy would be reading a signal that is not there"
    )


def test_a_constant_signal_leaves_the_mixture_untouched(four_sources):
    """The other half of the contract: identical input must not move weights.

    If per-source losses never differ, every source is at the mean, so every source
    is `sama` and the weights must be exactly what they were.
    """
    ds = SourceMixtureDataset(four_sources, STATIC)
    before = ds.probabilities()
    applied = ds.update_weights_from_signal({n: 3.0 for n in ds.source_names})
    assert set(applied.values()) == {SAMA}
    after = ds.probabilities()
    for k in before:
        assert after[k] == pytest.approx(before[k], abs=1e-9), k


def test_policy_responds_to_a_contrived_spread(four_sources):
    """Sanity: a real spread must move weights in the registered direction.

    Guards against a policy that no longer reads its input at all, which is the
    failure mode a constant-loss test cannot see.
    """
    ds = SourceMixtureDataset(four_sources, STATIC)
    start = ds.probabilities()
    # python is clearly the worst source, repeatedly.
    for _ in range(20):
        ds.update_weights_from_signal(
            {"fineweb": 1.0, "tinystories": 1.1, "cosmopedia": 1.2, "python": 8.0}
        )
    end = ds.probabilities()
    assert end["python"] < start["python"]
    assert end["fineweb"] > start["fineweb"]
    assert pytest.approx(sum(end.values()), abs=1e-6) == 1.0
