"""X25 Stage 1: in-place test-time training (inference-only).

Mechanism (frozen by docs/preregistrations/2026-10-05-X25-ttt-inference.md):
every ``chunk_tokens`` generated tokens, take ``ttt_steps`` gradient steps on
that chunk's own next-token loss, updating ONLY the per-block MLP
down-projection matrices (``mlp.down``). Everything else stays frozen.
Weights reset to base before every story — fast weights are story-local
memory, not permanent learning.

No training involved: this module only runs inference with interleaved
self-supervised updates. The LR=0 control runs the identical code path with
a zero step, so any measured difference is the update, not scaffolding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model import GPT

FAST_PARAM_SUFFIX = "mlp.down.weight"


def fast_parameters(model: GPT) -> list[torch.nn.Parameter]:
    """The registered fast-weight set: one down-projection per block."""
    params = [p for name, p in model.named_parameters() if name.endswith(FAST_PARAM_SUFFIX)]
    assert len(params) == model.cfg.n_layer, (
        f"expected {model.cfg.n_layer} fast weights, found {len(params)}"
    )
    return params


def snapshot_weights(model: GPT) -> dict[str, torch.Tensor]:
    """Bitwise snapshot of the fast weights (for per-story reset)."""
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if name.endswith(FAST_PARAM_SUFFIX)
    }


def restore_weights(model: GPT, snapshot: dict[str, torch.Tensor]) -> None:
    """Restore fast weights from a snapshot (no grad tracking)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(FAST_PARAM_SUFFIX):
                p.copy_(snapshot[name])


def ttt_step(
    model: GPT,
    chunk_ids: torch.Tensor,
    lr: float,
    steps: int = 1,
) -> float:
    """One self-supervised update on a chunk; returns the mean chunk loss.

    Runs the identical code path for lr=0 (mathematical no-op), so the
    zero-update control cannot differ by scaffolding.
    """
    fast = fast_parameters(model)
    for p in fast:
        if p.grad is not None:
            p.grad = None
    total = 0.0
    for _ in range(max(1, steps)):
        logits, _ = model(chunk_ids[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), chunk_ids[:, 1:].reshape(-1)
        )
        loss.backward()
        total += float(loss.detach())
        with torch.no_grad():
            for p in fast:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)
        for p in fast:
            p.grad = None
    return total / max(1, steps)


def generate_with_ttt(
    model: GPT,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    chunk_tokens: int = 256,
    ttt_lr: float = 1e-4,
    ttt_steps: int = 1,
    temperature: float = 0.8,
    top_k: int | None = 200,
    seed: int = 7,
    reset: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Generate with interleaved fast-weight updates.

    Returns (full token sequence, diagnostics with update count and losses).
    With reset=True (default), fast weights are restored to base on exit, so
    nothing leaks across stories.
    """
    snapshot = snapshot_weights(model)
    generator = torch.Generator(device=prompt_ids.device).manual_seed(seed)
    was_training = model.training
    model.eval()
    idx = prompt_ids.clone()
    update_losses: list[float] = []
    updates = 0
    try:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.cfg.context_len :]
            with torch.no_grad():
                logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                top = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < top, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat((idx, nxt), dim=1)
            generated = idx.size(1) - prompt_ids.size(1)
            if generated % chunk_tokens == 0:
                chunk = idx[:, -chunk_tokens:]
                update_losses.append(ttt_step(model, chunk, ttt_lr, ttt_steps))
                updates += 1
    finally:
        if reset:
            restore_weights(model, snapshot)
        if was_training:
            model.train()
    return idx, {"updates": updates, "update_losses": update_losses}
