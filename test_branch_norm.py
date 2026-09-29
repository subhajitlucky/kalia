"""Tests for X19's `branch_norm` arm.

X19 exists to attribute X18's −0.0436 nats, and X18's history is a run of
mechanisms that did not do what their code comments said: I15's document masking
was never wired in, and `GatedResidual`'s deliberate `zeros_` init was silently
overwritten by `GPT._init_weights`. G-2 and G-3 exist because an arm that is
silently misconfigured produces a clean, confident, wrong number — so both
correctness checks are asserted here, locally, before any GPU time is spent.

The headline claim is the parameter arithmetic. Amendment 1 to X19 corrects
G-2: `RMSNorm` has a learnable weight, so the branch norm is *not* free. Getting
that number wrong in the direction that made the arm sound better is exactly the
error the amendment documents, so it is pinned by test rather than by prose.
"""

from __future__ import annotations

import pytest
import torch
import yaml

from model import GPT, GPTConfig

MICRO = dict(vocab_size=50257, n_layer=6, n_head=6, n_embd=384, context_len=512)


def _cfg(flag: str) -> GPTConfig:
    return GPTConfig(**MICRO, **{flag: True})


def _params(flag: str | None) -> int:
    torch.manual_seed(0)
    cfg = GPTConfig(**MICRO) if flag is None else _cfg(flag)
    return GPT(cfg).num_params()


# --- G-2: parameter arithmetic ------------------------------------------------


def test_g2_branch_norm_parameter_count():
    """Amendment 1: 29,922,816, i.e. control + n_layer * n_embd."""
    control = _params(None)
    assert control == 29_920_512
    assert _params("branch_norm") == 29_922_816
    assert _params("branch_norm") - control == 2_304 == MICRO["n_layer"] * MICRO["n_embd"]


def test_g2_gate_contributes_exactly_its_own_matrices():
    """gated - branchnorm isolates the gate, which is the number that matters."""
    diff = _params("gated_residual") - _params("branch_norm")
    rank, dim, layers = 32, MICRO["n_embd"], MICRO["n_layer"]
    expected = layers * ((dim * rank + rank) + (rank * dim + dim))  # w1 + b1 + w2 + b2
    assert diff == 149_952
    assert diff == expected


def test_branch_norm_is_not_free_so_the_amendment_was_necessary():
    """If this fails, RMSNorm gained a non-affine mode and G-2 needs re-deriving."""
    assert _params("branch_norm") != _params(None), (
        "BranchNorm has become parameter-free; Amendment 1's arithmetic is stale"
    )


# --- G-3: the arm is a real change, not a near-identity ------------------------


def test_g3_branch_norm_rescales_every_quarter_to_unit_rms():
    torch.manual_seed(0)
    model = GPT(_cfg("branch_norm")).eval()

    # A stream with strongly unequal quarter-energies, which is the case the
    # branch norm is supposed to change.
    x = torch.randn(1, 8, MICRO["n_embd"])
    x[..., :96] *= 0.05
    x[..., 96:192] *= 12.0
    out = model.blocks[0].branch(x)

    quarters = [out[..., i * 96 : (i + 1) * 96] for i in range(4)]
    for i, q in enumerate(quarters):
        rms = q.pow(2).mean().sqrt().item()
        assert rms == pytest.approx(1.0, abs=0.02), f"quarter {i} RMS is {rms}"


def test_g3_differs_from_the_unnormalised_stream_by_well_over_one_percent():
    """The >1% bar is what makes this arm a real change rather than a no-op."""
    torch.manual_seed(0)
    model = GPT(_cfg("branch_norm")).eval()
    x = torch.randn(1, 8, MICRO["n_embd"])
    x[..., :96] *= 0.05
    x[..., 96:192] *= 12.0

    out = model.blocks[0].branch(x)
    raw_quarters = [x[..., i * 96 : (i + 1) * 96] for i in range(4)]
    out_quarters = [out[..., i * 96 : (i + 1) * 96] for i in range(4)]

    diffs = []
    for rq, oq in zip(raw_quarters, out_quarters):
        r = rq.pow(2).mean().sqrt().item()
        o = oq.pow(2).mean().sqrt().item()
        diffs.append(abs(o - r) / r)
    assert max(diffs) > 0.01, f"max relative RMS change was {max(diffs):.4%}"


# --- configuration integrity --------------------------------------------------


def test_the_two_arms_are_mutually_exclusive():
    with pytest.raises(AssertionError, match="alternative arms"):
        GPT(GPTConfig(**MICRO, gated_residual=True, branch_norm=True))


def test_config_files_differ_from_control_by_exactly_one_flag():
    """An ablation arm that quietly changes anything else is not an ablation."""
    base = yaml.safe_load(open("configs/micro-base.yaml"))
    gated = yaml.safe_load(open("configs/micro-gated.yaml"))
    branchnorm = yaml.safe_load(open("configs/micro-branchnorm.yaml"))

    for name, other, flag in (("gated", gated, "gated_residual"), ("branchnorm", branchnorm, "branch_norm")):
        assert other["model"][flag] is True
        assert "gated_residual" not in other["model"] or flag == "gated_residual"
        assert other["train"] == base["train"], f"{name} changed the training config"
        differing = {k for k in set(base["model"]) | set(other["model"])
                     if base["model"].get(k) != other["model"].get(k)}
        assert differing == {flag}, f"{name} differs from control in {differing}"


def test_seed_and_step_count_match_the_x18_arms():
    """X19 reuses X18's control and gated arms, so its arm must match their setup."""
    base = yaml.safe_load(open("configs/micro-base.yaml"))
    bn = yaml.safe_load(open("configs/micro-branchnorm.yaml"))
    gated = yaml.safe_load(open("configs/micro-gated.yaml"))
    for key in ("seed", "max_steps", "learning_rate", "warmup_steps", "optimizer"):
        assert bn["train"][key] == base["train"][key], key
        assert gated["train"][key] == base["train"][key], key
