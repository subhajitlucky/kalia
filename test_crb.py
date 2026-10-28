"""X38: correctness tests for the counterfactual residual decomposition."""

import pytest
import torch

from model import CRBDecomposition, GPT, GPTConfig


def tiny(**kw) -> GPTConfig:
    base = dict(
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=32,
        context_len=16,
        dropout=0.0,
    )
    base.update(kw)
    return GPTConfig(**base)


def test_default_off():
    assert GPTConfig().crb is False


def test_crb_rejects_memory_arm():
    with pytest.raises(AssertionError):
        GPT(tiny(crb=True, crb_stable_dim=8, memory_slots=4))


def test_rank_bounds():
    with pytest.raises(ValueError):
        CRBDecomposition(32, 0)
    with pytest.raises(ValueError):
        CRBDecomposition(32, 32)


def test_logits_match_control():
    """The decomposition is a reparameterisation: stable + volatile == delta."""
    torch.manual_seed(0)
    std = GPT(tiny()).eval()
    crb = GPT(tiny(crb=True, crb_stable_dim=8)).eval()
    crb.load_state_dict(std.state_dict(), strict=False)
    x = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        base_logits = std(x)[0]
        crb_logits = crb(x)[0]
    assert torch.allclose(base_logits, crb_logits, atol=1e-5)


def test_lm_loss_does_not_train_split():
    """The language path must be invariant to the decomposition by design."""
    torch.manual_seed(0)
    model = GPT(tiny(crb=True, crb_stable_dim=8))
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    for param in (model.crb.down.weight, model.crb.up.weight):
        assert param.grad is None or param.grad.abs().max() < 1e-8


def test_cf_loss_trains_split():
    torch.manual_seed(0)
    model = GPT(tiny(crb=True, crb_stable_dim=8))
    orig = torch.randint(0, 128, (4, 16))
    edit = torch.randint(0, 128, (4, 16))
    types = torch.tensor([0, 1, 0, 1])
    model.zero_grad(set_to_none=True)
    loss = model.counterfactual_loss(orig, edit, types)
    loss.backward()
    assert model.crb.up.weight.grad is not None
    assert model.crb.up.weight.grad.abs().max() > 0.0


def test_cf_loss_handles_one_type_only():
    torch.manual_seed(0)
    model = GPT(tiny(crb=True, crb_stable_dim=8))
    orig = torch.randint(0, 128, (4, 16))
    edit = torch.randint(0, 128, (4, 16))
    for types in (torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)):
        loss = model.counterfactual_loss(orig, edit, types)
        assert torch.isfinite(loss)


def test_pool_masks_padding():
    model = GPT(tiny())
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    mask = torch.tensor([[True, True, False, False]])
    pooled = model._pool(x, mask)
    assert torch.allclose(pooled, torch.tensor([[1.5]]))


def test_stable_state_shapes():
    torch.manual_seed(0)
    model = GPT(tiny(crb=True, crb_stable_dim=8))
    x = torch.randint(0, 128, (2, 16))
    z = model.stable_state(x)
    assert z.shape == (2, 32)
    stats = model.stable_stats(x)
    assert set(stats) == {"stable_norm", "effective_rank"}
    assert stats["stable_norm"] > 0.0
    assert stats["effective_rank"] >= 1.0


def test_non_crb_stable_state_is_hidden():
    torch.manual_seed(0)
    model = GPT(tiny())
    x = torch.randint(0, 128, (2, 16))
    assert model.stable_state(x).shape == (2, 32)
    assert model.stable_stats(x) == {}
