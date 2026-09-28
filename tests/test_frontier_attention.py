"""Tests for the two 2026 frontier changes: NoPE and Gated Residual.

NoPE   -- SmolLM3: drop rotary embeddings on every Nth layer.
Gated  -- Qwen3.8-Flash-Next: widen/gate the residual path, data-dependent read.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import GPT, GPTConfig  # noqa: E402

BASE = dict(vocab_size=512, n_layer=6, n_head=6, n_embd=96, context_len=64)


def test_nope_is_off_by_default():
    m = GPT(GPTConfig(**BASE))
    assert all(b.attn.use_rope for b in m.blocks)


def test_nope_drops_rope_on_the_right_layers():
    m = GPT(GPTConfig(**BASE, nope_interval=4))
    dropped = [i for i, b in enumerate(m.blocks) if not b.attn.use_rope]
    # every 4th layer, counting from 1: layers 4 and 8 -> indices 3 and 7
    assert dropped == [3]


def test_nope_is_parameter_neutral():
    """The point of NoPE is removing machinery, not adding or removing capacity."""
    a = GPT(GPTConfig(**BASE)).num_params()
    b = GPT(GPTConfig(**BASE, nope_interval=4)).num_params()
    assert a == b


def test_nope_changeless_config_leaves_layers_alone():
    m = GPT(GPTConfig(**BASE, nope_interval=0))
    assert all(b.attn.use_rope for b in m.blocks)


def test_nope_forward_runs_and_is_finite():
    m = GPT(GPTConfig(**BASE, nope_interval=4))
    ids = torch.randint(0, 512, (2, 16))
    _, loss = m(ids, ids)
    assert torch.isfinite(loss)


def test_gated_residual_is_off_by_default():
    m = GPT(GPTConfig(**BASE))
    assert all(b.gated is None for b in m.blocks)


def test_gated_starts_nearly_closed():
    """Initialising the gate wide open would mean the arm is not comparable to
    control at step 0; it has to start as a faithful copy of the pre-norm path."""
    m = GPT(GPTConfig(**BASE, gated_residual=True))
    g = m.blocks[0].gated
    gate = torch.sigmoid(g.w2(torch.nn.functional.silu(g.w1(torch.randn(1, BASE["n_embd"])))))
    assert gate.mean() < 0.05, f"gate should start near-closed, got {gate.mean():.3f}"


def test_gated_preserves_width_and_shape():
    m = GPT(GPTConfig(**BASE, gated_residual=True))
    ids = torch.randint(0, 512, (2, 16))
    logits, loss = m(ids, ids)
    assert logits.shape == (2, 16, BASE["vocab_size"])
    assert torch.isfinite(loss)


def test_gated_preserves_causality():
    """A residual path that could see the future would be a silent, fatal bug."""
    torch.manual_seed(0)
    m = GPT(GPTConfig(**BASE, gated_residual=True)).eval()
    ids = torch.randint(0, 512, (1, 24))
    a, _ = m(ids)
    ids2 = ids.clone()
    ids2[0, 12:] = torch.randint(0, 512, (12,), generator=torch.Generator().manual_seed(1))
    b, _ = m(ids2)
    assert torch.allclose(a[0, :12], b[0, :12], atol=1e-5), "future tokens leaked into the past"


def test_gated_adds_only_the_low_rank_gate():
    control = GPT(GPTConfig(**BASE)).num_params()
    gated = GPT(GPTConfig(**BASE, gated_residual=True)).num_params()
    added = gated - control
    d, r, nb = BASE["n_embd"], 32, 4
    # per block: n_branch norms of d/nb, plus w1 (d->r) and w2 (r->d) with biases
    per_block = nb * (d // nb) + (d * r + r) + (r * d + d)
    assert added == per_block * BASE["n_layer"], "unexpected parameter delta"


def test_gated_is_cheap_at_the_dimensions_we_actually_use():
    """The <2% claim is about the real 384-dim micro model, not the tiny fixture
    above, where a rank-32 gate is proportionally far more expensive."""
    real = dict(vocab_size=50257, n_layer=6, n_head=6, n_embd=384, context_len=512)
    control = GPT(GPTConfig(**real)).num_params()
    gated = GPT(GPTConfig(**real, gated_residual=True)).num_params()
    added = gated - control
    assert added < control * 0.01, f"gate added {added} on a {control} model"


def test_gated_backward_reaches_the_gate():
    """A gate that is constructed but receives no gradient would pass every test above."""
    m = GPT(GPTConfig(**BASE, gated_residual=True))
    ids = torch.randint(0, 512, (2, 16))
    _, loss = m(ids, ids)
    loss.backward()
    g = m.blocks[0].gated
    assert g.w2.weight.grad is not None
    assert g.w2.weight.grad.abs().sum() > 0, "gate received no gradient signal"


@pytest.mark.parametrize("cfg", [
    {"nope_interval": 4},
    {"gated_residual": True},
    {"nope_interval": 4, "gated_residual": True},
])
def test_variants_construct_and_step(cfg):
    m = GPT(GPTConfig(**BASE, **cfg))
    ids = torch.randint(0, 512, (2, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    before = m(ids, ids)[1].item()
    for _ in range(10):  # a single step on a random model can legitimately overshoot
        opt.zero_grad()
        m(ids, ids)[1].backward()
        opt.step()
    assert m(ids, ids)[1].item() < before, "loss did not decrease over 10 steps"
