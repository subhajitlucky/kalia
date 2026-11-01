"""X40: model-side correctness tests for the event workspace.

Before any GPU time, three things must be true:

1. **The event arm starts as its control.** The read projection is zero, so
   every base logit is identical at initialisation. Otherwise the comparison
   measures a different starting model.
2. **The workspace is causal.** A later chunk's tokens cannot change an
   earlier chunk's logits or the prompt-boundary workspace.
3. **The IP loss trains the workspace.** The auxiliary next-state loss must
   reach the write gate, the write values and the intervention head, or the
   mechanism is not actually being run.
"""

from __future__ import annotations

import torch

from intervention_pairs import InterventionDataset, write_pairs
from model import GPT, GPTConfig

TINY = dict(
    vocab_size=50257, n_layer=2, n_head=2, n_embd=32, context_len=64,
    dropout=0.0, qk_norm=True, logit_softcap=30.0,
)
EVENT = dict(
    event_slots=4, event_slot_dim=8, event_chunk=48, event_head=True,
)


def tiny(**extra):
    return GPT(GPTConfig(**TINY, **extra))


def pair_batch(tmp_path, batch_size=4):
    write_pairs(tmp_path, seed=7, n_train=30, n_eval=10, n_comp=10)
    ds = InterventionDataset(tmp_path / "train.jsonl", prefix_len=48, suffix_len=8)
    return ds.get_batch(
        batch_size, torch.device("cpu"), torch.Generator().manual_seed(0)
    )


def test_event_model_starts_as_control():
    torch.manual_seed(0)
    control = tiny()
    torch.manual_seed(0)
    event = tiny(**EVENT)
    for (name, p), (ename, q) in zip(
        control.named_parameters(), event.named_parameters()
    ):
        if name.startswith(("event.", "intervention_head.")):
            continue
        assert ename == name
        assert torch.equal(p, q), f"{name} diverged at initialisation"
    x = torch.randint(0, 50257, (2, 64))
    control.eval()
    event.eval()
    with torch.no_grad():
        a, _ = control(x)
        b, _ = event(x)
    assert torch.equal(a, b), "the zero-init read did not preserve the control path"


def test_event_forward_backward_and_gradients(tmp_path):
    model = tiny(**EVENT)
    batch = pair_batch(tmp_path)
    loss, stats = model.ip_loss(batch, state_weight=0.1, div_weight=0.01)
    assert torch.isfinite(loss)
    assert set(stats) == {"lm", "state", "div"}
    loss.backward()
    named = dict(model.named_parameters())
    for name in (
        "event.write_gate.0.weight",
        "event.write_value.weight",
        "event.read_o.weight",
        "intervention_head.ws_proj.weight",
        "intervention_head.net.0.weight",
    ):
        grad = named[name].grad
        assert grad is not None and grad.abs().sum() > 0, f"{name} got no gradient"


def test_workspace_is_causal(tmp_path):
    model = tiny(**EVENT)
    model.eval()
    # Open the read so the causal path is actually exercised, not the
    # zero-init shortcut.
    with torch.no_grad():
        torch.nn.init.normal_(model.event.read_o.weight, std=0.05)
    batch = pair_batch(tmp_path, batch_size=2)
    x = batch.x.clone()
    with torch.no_grad():
        _, _, before = model(
            x, attn_mask=batch.attn_mask, token_mask=batch.token_mask,
            return_events=True,
        )
        changed = x.clone()
        changed[:, 48 + 3] = (changed[:, 48 + 3] + 1) % 50257
        _, _, after = model(
            changed, attn_mask=batch.attn_mask, token_mask=batch.token_mask,
            return_events=True,
        )
    # The prompt chunk (positions 0..47) cannot see a suffix token.
    assert torch.equal(before["workspace"], after["workspace"])
    assert before["boundary"].shape == (2, 32)
    # A real prompt token can change the prompt-boundary workspace.
    x2 = x.clone()
    pos = int(batch.token_mask[0, :48].nonzero()[0])
    x2[:, pos] = (x2[:, pos] + 1) % 50257
    with torch.no_grad():
        _, _, shifted = model(
            x2, attn_mask=batch.attn_mask, token_mask=batch.token_mask,
            return_events=True,
        )
    assert not torch.equal(before["workspace"], shifted["workspace"])


def test_event_forward_without_targets_and_generate(tmp_path):
    model = tiny(**EVENT)
    model.eval()
    batch = pair_batch(tmp_path, batch_size=2)
    with torch.no_grad():
        logits, loss = model(batch.x, attn_mask=batch.attn_mask)
    assert logits.shape == (2, 56, 50257)
    assert loss is None
    out = model.generate(batch.x[:1, :12], max_new_tokens=2)
    assert out.shape == (1, 14)


def test_ip_loss_requires_the_head(tmp_path):
    model = tiny(event_slots=4, event_slot_dim=8, event_chunk=48)
    batch = pair_batch(tmp_path)
    try:
        model.ip_loss(batch)
    except ValueError as err:
        assert "event_head" in str(err)
    else:
        raise AssertionError("workspace-only model accepted ip_loss")


def test_adapter_matches_the_workspace_parameter_count():
    dim = TINY["n_embd"]
    event = tiny(**EVENT)
    event_params = sum(p.numel() for p in event.event.parameters())
    hidden = max(1, round(event_params / (2 * dim)))
    adapter = tiny(memory_control=True, memory_control_hidden=hidden)
    adapter_params = sum(p.numel() for p in adapter.adapter.parameters())
    # One hidden unit is the finest match the shared OutputAdapter allows.
    assert abs(adapter_params - event_params) <= 2 * dim


def test_event_and_other_arms_do_not_combine():
    for extra in (
        dict(event_slots=2, memory_slots=2),
        dict(event_slots=2, memory_control=True),
        dict(event_slots=2, crb=True),
        dict(event_slots=2, sutra=True),
        dict(event_slots=2, event_head=True, crb=True),
    ):
        try:
            tiny(**extra)
        except AssertionError:
            continue
        raise AssertionError(f"combined arms were accepted: {extra}")
