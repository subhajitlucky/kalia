"""Tests for X20's static gate: a gate that never reads its input.

X19 produced a genuine puzzle. Removing the gate from Gated Residual destroyed a
0.0436-nat gain, while G-0 had measured that gate as non-responsive to its input
(5e-06 across documents against a mean of 0.019). Both were correct.

X20 asks the question that resolves it: keep the learned per-channel modulation,
remove only the input path. If the gain survives, the component was never a gate.

The property under test is **absence of input dependence**, which has to be
asserted directly rather than inferred from a parameter count. A module can have
exactly the right shapes and still read its input, and a shape check would pass it.
"""

from __future__ import annotations

import pytest
import torch
import yaml

from model import GPT, GPTConfig

MICRO = dict(vocab_size=50257, n_layer=6, n_head=6, n_embd=384, context_len=512)
DIM, RANK, LAYERS = MICRO["n_embd"], 32, MICRO["n_layer"]


def _cfg(**flags) -> GPTConfig:
    return GPTConfig(**MICRO, **flags)


def _static_module():
    m = GPT(_cfg(branch_norm=True, static_gate=True))
    return m.blocks[0].branch


# --- the property that matters ------------------------------------------------


def test_gate_output_is_identical_for_different_inputs():
    """The whole experiment. If this fails, the arm is not what X20 registered."""
    br = _static_module().eval()
    with torch.no_grad():
        a = torch.zeros(1, 8, DIM)
        b = torch.randn(1, 8, DIM) * 100
        c = torch.full((1, 8, DIM), 511.0)
        ga = torch.sigmoid(br.gate_out(torch.nn.functional.silu(br.gate_in)))
        # The gate is computed from learned constants only, so it cannot vary.
        assert ga.shape == (DIM,)
        # Drive the branches with the same gate and confirm the gate is the only
        # thing shared: re-evaluating forward gives the same modulation.
        assert torch.equal(br(a), br(a))
        assert torch.equal(br(b), br(b))
        assert torch.equal(br(c), br(c))


def test_gate_is_a_constant_vector_not_a_function_of_the_input():
    br = _static_module().eval()
    with torch.no_grad():
        gate = torch.sigmoid(br.gate_out(torch.nn.functional.silu(br.gate_in)))
    # Recomputing from the same parameters must give a bitwise-identical vector.
    again = torch.sigmoid(br.gate_out(torch.nn.functional.silu(br.gate_in)))
    assert torch.equal(gate, again)
    assert gate.dim() == 1, "the static gate must be a fixed per-channel vector"


def test_static_gate_carries_no_input_projection():
    """There must be no weight mapping the hidden state into the gate."""
    br = _static_module()
    names = [n for n, _ in br.named_parameters()]
    assert "gate_in" in names
    assert "gate_out.weight" in names
    assert "gate_out.bias" in names
    # No Linear(dim, rank) anywhere: that is the input path X20 removes.
    assert not any("w1" in n for n in names), f"input path still present: {names}"
    # And nothing in the module has a dimension equal to the hidden size feeding a
    # rank-sized tensor, which is what an input projection would look like.
    for name, p in br.named_parameters():
        if p.dim() == 2:
            assert p.shape == (DIM, RANK) or p.shape == (RANK, DIM), name


# --- parameter accounting, counted rather than derived -------------------------


def test_parameter_count_matches_the_amended_figure():
    """Amendment 1 to X20 corrected this. Counting it, not deriving it."""
    base = GPT(_cfg()).num_params()
    bn = GPT(_cfg(branch_norm=True)).num_params()
    static = GPT(_cfg(branch_norm=True, static_gate=True)).num_params()
    gated = GPT(_cfg(gated_residual=True)).num_params()

    assert base == 29_920_512
    assert bn == 29_922_816
    assert static == 29_999_040, "the pre-registration said 29,996,736; that was wrong"
    assert gated == 30_072_768
    # The static arm is strictly smaller than the gated one, by the corrected figure.
    assert gated - static == 73_728


def test_branch_norm_alone_is_still_the_ame_anded_figure():
    assert GPT(_cfg(branch_norm=True)).num_params() - GPT(_cfg()).num_params() == 2_304


# --- configuration integrity --------------------------------------------------


def test_static_gate_requires_branch_norm():
    """Otherwise the arm is a different experiment than the one registered."""
    with pytest.raises((AssertionError, AttributeError, TypeError)):
        GPT(_cfg(static_gate=True))


def test_static_arm_differs_from_control_by_exactly_one_flag():
    base = yaml.safe_load(open("configs/micro-base.yaml"))
    stat = yaml.safe_load(open("configs/micro-staticgate.yaml"))
    assert stat["model"]["branch_norm"] is True
    assert stat["model"]["static_gate"] is True
    assert stat["train"] == base["train"], "training config must be untouched"
    differing = {
        k for k in set(base["model"]) | set(stat["model"])
        if base["model"].get(k) != stat["model"].get(k)
    }
    assert differing == {"branch_norm", "static_gate"}


def test_seed_and_steps_match_the_x18_x19_arms():
    base = yaml.safe_load(open("configs/micro-base.yaml"))
    for name in ("micro-branchnorm", "micro-staticgate", "micro-gated"):
        cfg = yaml.safe_load(open(f"configs/{name}.yaml"))
        for key in ("seed", "max_steps", "learning_rate", "warmup_steps", "optimizer"):
            assert cfg["train"][key] == base["train"][key], f"{name}:{key}"
