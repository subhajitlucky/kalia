"""X37 SUTRA: the rule grammar must be causal and parameter-fair before GPU time.

Two failure modes would make the experiment meaningless:

1. **Future leakage** through the adhikāra scope or anuvṛtti carry. The scope is
   the *previous* chunk's mean; the carry is updated only at a chunk end and read
   only by the next chunk.
2. **A capacity confound**. The SUTRA block adds a rule codebook and a router, so
   the control is a wider SwiGLU matched to its parameter count, not the standard
   micro MLP.
"""

from __future__ import annotations

import yaml

import pytest
import torch

from model import GPT, GPTConfig

TINY = dict(
    vocab_size=256, n_layer=2, n_head=2, n_embd=32, context_len=16,
    dropout=0.0, qk_norm=True,
)


def tiny(**extra):
    return GPT(GPTConfig(**TINY, **extra))


def sutra(**extra):
    return tiny(
        sutra=True, sutra_rules=8, sutra_rank=2, sutra_general_hidden=64,
        sutra_chunk=4, sutra_carry_dim=8, **extra,
    )


def test_sutra_forward_backward_and_usage():
    m = sutra(sutra_exception=True)
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    logits, loss = m(x, y)
    assert logits.shape == (2, 16, 256)
    assert torch.isfinite(loss)
    loss.backward()
    mlp = m.blocks[0].mlp
    assert mlp.last_usage.shape == (8,)
    assert torch.allclose(mlp.last_usage.sum(), torch.tensor(1.0), atol=1e-5)
    assert mlp.last_probs.shape == (8,)
    assert torch.allclose(mlp.last_probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.isfinite(mlp.aux_loss)
    assert mlp.rules_A.grad is not None and mlp.rules_A.grad.abs().sum() > 0
    assert mlp.rules_B.grad is not None and mlp.rules_B.grad.abs().sum() > 0
    assert mlp.router.weight.grad is not None and mlp.router.weight.grad.abs().sum() > 0
    out = m.generate(x[:1, :8], max_new_tokens=4)
    assert out.shape == (1, 12)


def test_sutra_is_causal():
    m = sutra()
    m.eval()
    x = torch.randint(0, 256, (1, 16))
    x2 = x.clone()
    x2[:, 12:] = torch.randint(0, 256, (1, 4))
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(x2)
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-6)
    assert not torch.allclose(a[:, 12:], b[:, 12:], atol=1e-6)


def test_sutra_control_is_parameter_matched():
    base = yaml.safe_load(open("configs/micro-muon.yaml"))["model"]
    base["qk_norm"] = True
    base["logit_softcap"] = 0.0
    control = GPT(GPTConfig(**base, sutra_control=True, sutra_control_hidden=1408))
    full = GPT(GPTConfig(
        **base, sutra=True, sutra_rules=64, sutra_rank=8,
        sutra_general_hidden=1024, sutra_chunk=64, sutra_carry_dim=64,
        sutra_carry=True, sutra_scope=True,
    ))
    exception = GPT(GPTConfig(
        **base, sutra=True, sutra_rules=64, sutra_rank=8,
        sutra_general_hidden=1024, sutra_chunk=64, sutra_carry_dim=64,
        sutra_carry=True, sutra_scope=True, sutra_exception=True,
    ))
    assert abs(control.num_params() - full.num_params()) / control.num_params() < 0.01
    assert abs(control.num_params() - exception.num_params()) / control.num_params() < 0.01
    # The standard micro anchor is smaller; the report must say so.
    assert GPT(GPTConfig(**base)).num_params() < control.num_params()


def test_sutra_components_can_be_switched_off():
    for extra in (
        dict(sutra_carry=False, sutra_scope=False),
        dict(sutra_carry=True, sutra_scope=False),
        dict(sutra_carry=False, sutra_scope=True),
    ):
        m = sutra(**extra)
        x = torch.randint(0, 256, (1, 16))
        logits, _ = m(x)
        assert torch.isfinite(logits).all()


def test_sutra_exception_starts_closed():
    m = sutra(sutra_exception=True)
    gate = m.blocks[0].mlp.exc_gate
    assert gate.bias.item() == pytest.approx(-4.0)


def test_sutra_top_k_is_one_for_now():
    with pytest.raises(AssertionError):
        tiny(sutra=True, sutra_rules=4, sutra_rank=2, sutra_general_hidden=64,
             sutra_chunk=4, sutra_carry_dim=8, sutra_top_k=2)


def test_sutra_and_memory_do_not_combine():
    with pytest.raises(AssertionError):
        tiny(sutra=True, sutra_rules=4, sutra_rank=2, sutra_general_hidden=64,
             sutra_chunk=4, sutra_carry_dim=8, memory_slots=4)
