"""X36 entity-memory model: correctness before any GPU time.

The memory model changes the output path, so the two things that can silently
ruin the experiment are:

1. **Future leakage** — a chunk's write must not affect an earlier token's
   logits, and the copy head must only see the causal prefix.
2. **A capacity confound** — the treatment and the control must carry the same
   number of trainable parameters; otherwise a win is just a bigger model.

These tests pin both, plus the recurrence labels and the copy scatter.
"""

from __future__ import annotations

import yaml

import torch

from model import GPT, GPTConfig

TINY = dict(
    vocab_size=256, n_layer=2, n_head=2, n_embd=32, context_len=16,
    dropout=0.0, qk_norm=True,
)


def tiny(**extra):
    return GPT(GPTConfig(**TINY, **extra))


def test_memory_forward_backward_and_generation():
    m = tiny(memory_slots=4, memory_chunk=4, memory_copy=True, recurrence_weight=0.1)
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    logits, loss = m(x, y)
    assert logits.shape == (2, 16, 256)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.memory.parameters()
    ), "the workspace received no gradient"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.copy.parameters()
    ), "the copy head received no gradient"
    out = m.generate(x[:1, :8], max_new_tokens=4)
    assert out.shape == (1, 12)


def test_workspace_is_causal():
    """Changing tokens in the last chunk must not move earlier logits."""
    m = tiny(memory_slots=4, memory_chunk=4, memory_copy=True)
    m.eval()
    x = torch.randint(0, 256, (1, 16))
    x2 = x.clone()
    x2[:, 12:] = torch.randint(0, 256, (1, 4))
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(x2)
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-6)
    assert not torch.allclose(a[:, 12:], b[:, 12:], atol=1e-6)


def test_copy_head_is_causal():
    """The copy distribution at an earlier position must ignore future tokens."""
    m = tiny(memory_copy=True)
    m.eval()
    x = torch.randint(0, 256, (1, 16))
    x2 = x.clone()
    x2[:, 12:] = torch.randint(0, 256, (1, 4))
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(x2)
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-6)


def test_recurrence_labels_are_self_supervised_and_causal():
    m = tiny(memory_slots=4, memory_chunk=4, recurrence_window=3)
    idx = torch.tensor([[1, 2, 3, 2, 5, 6, 2, 8]])
    labels, mask = m._recurrence_labels(idx)
    # position 1 (token 2) recurs at 3; position 3 (token 2) recurs at 6;
    # position 5 (token 6) recurs at 6? no -- 6 is the next token id, and 6
    # appears only once. Labels are "same id appears again", not "value >".
    assert labels[0, 1] and labels[0, 3]
    assert not labels[0, 6]
    assert not labels[0, 7]
    assert mask.dtype == torch.bool


def test_copy_scatter_only_uses_prefix_ids():
    m = tiny(memory_copy=True)
    m.eval()
    x = torch.tensor([[5, 7, 5, 9]])
    h = torch.randn(1, 4, 32)
    with torch.no_grad():
        copy = m.copy.distribution(h, h, x, 0)
    present = set(x[0].tolist())
    for v in range(256):
        if copy[..., v].max() > 1e-8:
            assert v in present, v
    assert torch.allclose(copy.sum(dim=-1), torch.ones(1, 4), atol=1e-5)


def test_micro_memory_matches_control_parameter_count():
    base = yaml.safe_load(open("configs/micro-qk-s1338.yaml"))["model"]
    control = GPT(GPTConfig(**base, memory_control=True, memory_control_hidden=1024))
    memory = GPT(GPTConfig(
        **base, memory_slots=32, memory_chunk=64, memory_copy=True,
        recurrence_weight=0.1,
    ))
    assert abs(control.num_params() - memory.num_params()) / control.num_params() < 0.001, (
        control.num_params(), memory.num_params()
    )


def test_memory_starts_near_its_control():
    """The read is zero-init and the write/copy gates start closed; the model
    must not begin with a large random memory contribution."""
    base = yaml.safe_load(open("configs/micro-qk-s1338.yaml"))["model"]
    torch.manual_seed(7)
    control = GPT(GPTConfig(**base, memory_control=True, memory_control_hidden=1024))
    torch.manual_seed(7)
    memory = GPT(GPTConfig(
        **base, memory_slots=32, memory_chunk=64, memory_copy=True,
        recurrence_weight=0.1,
    ))
    x = torch.randint(0, 50257, (1, 128))
    with torch.no_grad():
        a, _ = control(x)
        b, _ = memory(x)
    # Same base weights by construction; the copy head is a per-token constant
    # shift on absent ids, so compare the distributions, not the raw logits.
    la = torch.log_softmax(a, dim=-1)
    lb = torch.log_softmax(b, dim=-1)
    assert (la - lb).abs().max() < 0.5, (la - lb).abs().max()
