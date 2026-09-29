"""Tests for the X18 F-4 gate-opening measurement.

Two things are being pinned here.

1. The measurement must reproduce a value we already know. At initialisation
   ``w2.bias`` is exactly -4, so ``sigmoid(-4) == 0.0179862...`` is the floor of
   the gate regardless of the weight. Checking the measured mean against that
   closed form catches a mis-wired forward pass -- the failure mode that would
   otherwise make a real result look like a null.

2. The gate's documented initialisation does **not** survive model construction.
   ``GatedResidual.__init__`` deliberately sets ``w2.weight`` to zero, but
   ``GPT.__init__`` afterwards calls ``self.apply(self._init_weights)``, which
   re-runs ``nn.init.normal_`` over every ``nn.Linear`` and overwrites it. X18
   measured the consequence: the gate never opened (mean 0.0192 after 500 steps
   against a 0.05 bar), so the improvement it showed is attributable to the
   branch normalisation rather than to the gate. These tests lock the actual
   behaviour in so the bug cannot be quietly "fixed" later without someone
   noticing that the measurement basis changed. The fix belongs to X19, not here.
"""

from __future__ import annotations

import math

import pytest
import torch

from gate_probe import F4_THRESHOLD, INIT_GATE_MEAN, gate_means, gate_stats
from model import GPT, GPTConfig

SIGMOID_NEG_FOUR = 1.0 / (1.0 + math.exp(4.0))


def _gated_config(**overrides) -> GPTConfig:
    base = dict(
        vocab_size=512,
        n_embd=64,
        n_layer=3,
        n_head=4,
        context_len=32,
        gated_residual=True,
    )
    base.update(overrides)
    return GPTConfig(**base)


def test_measured_gate_matches_closed_form_floor_at_init():
    """A fresh model reads sigmoid(-4) per element, so the mean must too."""
    torch.manual_seed(0)
    model = GPT(_gated_config()).eval()
    batches = [torch.randint(0, 512, (1, 16)), torch.randint(0, 512, (2, 8))]

    result = gate_means(model, batches)

    assert SIGMOID_NEG_FOUR == INIT_GATE_MEAN
    assert len(result["per_block"]) == 3
    assert result["gated_blocks"] == 3
    for value in result["per_block"]:
        assert value == pytest.approx(SIGMOID_NEG_FOUR, rel=1e-3)
    assert result["overall"] == pytest.approx(SIGMOID_NEG_FOUR, rel=1e-3)


def test_gate_weight_init_is_overwritten_by_global_reinit():
    """Documents the bug: the deliberate zeros_ never survive GPT construction.

    If this starts failing, the init has been repaired and the X18 measurement
    basis has changed -- which invalidates comparing any new run against the
    0.0192 that was recorded for X18.
    """
    torch.manual_seed(0)
    model = GPT(_gated_config())

    for block in model.blocks:
        assert block.gated.w2.bias.detach().mean().item() == pytest.approx(-4.0, abs=1e-6)
        # GatedResidual.__init__ sets this to exactly zero. It does not stay zero.
        assert block.gated.w2.weight.detach().abs().mean().item() > 1e-3


def test_measurement_tracks_the_weights_rather_than_a_constant():
    """Guards against a detached or hard-coded reading that training cannot move.

    Nudging the bias must move the measurement by the amount sigmoid predicts.
    """
    torch.manual_seed(0)
    model = GPT(_gated_config()).eval()
    batches = [torch.randint(0, 512, (1, 16))]

    before = gate_means(model, batches)["overall"]

    with torch.no_grad():
        for block in model.blocks:
            block.gated.w2.bias += 2.0  # sigmoid(-2) ~ 0.1192

    after = gate_means(model, batches)["overall"]

    assert after > before
    assert after == pytest.approx(1.0 / (1.0 + math.exp(2.0)), rel=1e-2)
    assert after > F4_THRESHOLD, "a +2.0 bias shift should open the gate past the bar"


def test_dispersion_distinguishes_a_constant_gate_from_a_data_dependent_one():
    """G-0's whole purpose: prove the statistic can tell inert from data-dependent.

    X18 concluded the gate was inert from its *mean* (0.0192). That inference is
    only valid if the measure can detect a low-mean-but-variable gate, so this
    builds both cases explicitly and checks they are separable.
    """
    torch.manual_seed(0)
    model = GPT(_gated_config()).eval()
    batches = [torch.randint(0, 512, (1, 16))]

    # Case 1: gate forced constant, w2.weight = 0, so the pre-activation is the
    # bias alone and every channel gets an identical value.
    with torch.no_grad():
        for block in model.blocks:
            block.gated.w2.weight.zero_()
    const = gate_stats(model, batches)

    # Case 2: w2.weight given real variation, so channels get different values.
    with torch.no_grad():
        for block in model.blocks:
            block.gated.w2.weight.normal_(0.0, 0.5)
    varied = gate_stats(model, batches)

    # Same-ish mean, very different dispersion -- the whole point.
    assert const["max_std_channel"] < 1e-6, "a zeroed w2 must give zero channel spread"
    assert varied["max_std_channel"] > 1e-3, "varying w2 must show channel spread"
    assert varied["max_std_channel"] > const["max_std_channel"] * 100


def test_dispersion_detects_input_dependence():
    """std_input must move when the gate actually reads its input."""
    torch.manual_seed(0)
    model = GPT(_gated_config()).eval()
    # Two very different inputs, so a data-dependent gate must separate them.
    batches = [
        torch.zeros(1, 16, dtype=torch.long),
        torch.full((1, 16), 511, dtype=torch.long),
    ]

    # Scale both matrices up. The gate is a rank-32 product of two small
    # matrices, so at init the input signal is ~1e-5 -- real, but far too weak
    # to demonstrate that the statistic can see input dependence at all. The
    # test is about the measure, so give it a gate that is unambiguously
    # data-dependent.
    with torch.no_grad():
        for block in model.blocks:
            block.gated.w1.weight.normal_(0.0, 2.0)
            block.gated.w2.weight.normal_(0.0, 2.0)
    varied = gate_stats(model, batches)

    with torch.no_grad():
        for block in model.blocks:
            block.gated.w1.weight.zero_()
            block.gated.w2.weight.zero_()
    constant = gate_stats(model, batches)

    assert varied["max_std_input"] > 1e-3, "a real gate must vary with its input"
    assert constant["max_std_input"] < 1e-9, "a bias-only gate cannot vary with input"
    assert varied["max_std_input"] > constant["max_std_input"] * 1000


def test_ungated_model_is_rejected_rather_than_silently_zero():
    """Scoring a non-gated checkpoint must fail loudly, not report 0.0."""
    torch.manual_seed(0)
    model = GPT(_gated_config(gated_residual=False)).eval()
    with pytest.raises(RuntimeError, match="no GatedResidual"):
        gate_means(model, [torch.randint(0, 512, (1, 16))])
