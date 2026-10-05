"""Unit tests for ttt.py (X25 Stage 1 mechanism).

All run on CPU against a tiny random-init model — no checkpoint, no GPU.
They assert the registered mechanism properties: fast-weight set, per-story
reset, LR=0 no-op, update cadence.
"""

from __future__ import annotations

import torch

from model import GPT, GPTConfig
from ttt import (
    FAST_PARAM_SUFFIX,
    fast_parameters,
    generate_with_ttt,
    restore_weights,
    snapshot_weights,
    ttt_step,
)


def tiny_model() -> GPT:
    cfg = GPTConfig(
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        context_len=64,
        qk_norm=False,
        logit_softcap=0.0,
    )
    torch.manual_seed(0)
    return GPT(cfg)


def test_fast_weights_are_exactly_the_mlp_down_projections():
    model = tiny_model()
    fast = fast_parameters(model)
    assert len(fast) == 2
    names = [n for n, _ in model.named_parameters() if n.endswith(FAST_PARAM_SUFFIX)]
    assert len(names) == 2
    assert all(f is p for f, p in zip(fast, [model.blocks[i].mlp.down.weight for i in range(2)]))


def test_snapshot_restore_is_bitwise():
    model = tiny_model()
    before = snapshot_weights(model)
    ttt_step(model, torch.randint(0, 256, (1, 16)), lr=1e-4)
    changed = any(
        not torch.equal(p.detach(), before[n])
        for n, p in model.named_parameters()
        if n.endswith(FAST_PARAM_SUFFIX)
    )
    assert changed, "an update must move the fast weights"
    restore_weights(model, before)
    for n, p in model.named_parameters():
        if n.endswith(FAST_PARAM_SUFFIX):
            assert torch.equal(p.detach(), before[n])


def test_lr_zero_leaves_weights_untouched():
    model = tiny_model()
    before = snapshot_weights(model)
    loss = ttt_step(model, torch.randint(0, 256, (1, 16)), lr=0.0)
    assert loss > 0
    for n, p in model.named_parameters():
        if n.endswith(FAST_PARAM_SUFFIX):
            assert torch.equal(p.detach(), before[n])


def test_non_fast_weights_never_move():
    model = tiny_model()
    frozen = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if not n.endswith(FAST_PARAM_SUFFIX)
    }
    ttt_step(model, torch.randint(0, 256, (1, 16)), lr=1e-2)
    for n, p in model.named_parameters():
        if not n.endswith(FAST_PARAM_SUFFIX):
            assert torch.equal(p.detach(), frozen[n]), n


def test_generate_runs_cadence_and_resets():
    model = tiny_model()
    before = snapshot_weights(model)
    prompt = torch.randint(0, 256, (1, 8))
    out, diag = generate_with_ttt(
        model, prompt, max_new_tokens=20, chunk_tokens=5, ttt_lr=1e-4,
        temperature=1.0, top_k=10, seed=7,
    )
    assert out.size(1) == 8 + 20
    assert diag["updates"] == 4
    assert len(diag["update_losses"]) == 4
    for n, p in model.named_parameters():
        if n.endswith(FAST_PARAM_SUFFIX):
            assert torch.equal(p.detach(), before[n])


def test_generate_is_deterministic_for_fixed_seed():
    prompt = torch.randint(0, 256, (1, 8))
    kwargs = dict(max_new_tokens=12, chunk_tokens=5, ttt_lr=1e-4,
                  temperature=1.0, top_k=10, seed=7)
    out1, _ = generate_with_ttt(tiny_model(), prompt, **kwargs)
    out2, _ = generate_with_ttt(tiny_model(), prompt, **kwargs)
    assert torch.equal(out1, out2)
