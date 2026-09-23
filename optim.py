"""Optimizers for KALIA.

Muon orthogonalizes the momentum update of 2D hidden weight matrices through a
quintic Newton-Schulz iteration, then steps along the resulting matrix. The
token embedding, the LM head and every 1D parameter stay on AdamW. Based on the
nanoGPT speedrun recipe (Jordan et al.) and used at scale by Kimi K2 and GLM-4.5.
"""

import torch
from torch.optim import AdamW, Optimizer


def zeroth_power_via_newtonschulz5(
    g: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Approximate the orthogonal polar factor of a 2D matrix ``g``."""
    assert g.ndim == 2, "Newton-Schulz orthogonalization expects a 2D matrix"
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.float() / (g.norm() + eps)
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


def normalize_update(x: torch.Tensor, direction: str, eps: float = 1e-8) -> torch.Tensor:
    """Muon+ post-polar normalization: unit-norm columns and/or rows.

    Directions: "none", "col", "row", "col_row", "row_col" (arXiv 2602.21545).
    """
    if direction in ("none", ""):
        return x
    if direction == "col":
        return x / (x.norm(dim=0, keepdim=True) + eps)
    if direction == "row":
        return x / (x.norm(dim=1, keepdim=True) + eps)
    if direction == "col_row":
        x = x / (x.norm(dim=0, keepdim=True) + eps)
        return x / (x.norm(dim=1, keepdim=True) + eps)
    if direction == "row_col":
        x = x / (x.norm(dim=1, keepdim=True) + eps)
        return x / (x.norm(dim=0, keepdim=True) + eps)
    raise ValueError(f"unknown normalization direction: {direction}")


class Muon(Optimizer):
    """Momentum SGD whose update is orthogonalized by Newton-Schulz."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        muon_plus: str = "none",
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            muon_plus=muon_plus,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad, dtype=torch.float32)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(grad.float())
                if group["nesterov"]:
                    update = grad.float().add(buf, alpha=group["momentum"])
                else:
                    update = buf
                update = zeroth_power_via_newtonschulz5(update, steps=group["ns_steps"])
                update = normalize_update(update, group["muon_plus"])
                update = update * (max(1.0, p.size(0) / p.size(1)) ** 0.5)
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"])
        return loss


def split_muon_params(model) -> tuple[list, list]:
    """Split parameters into (hidden 2D weights for Muon, everything else).

    Tied weights (the token embedding is the LM head) are de-duplicated so the
    shared tensor is optimized exactly once.
    """
    hidden, other, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_embed_or_head = name.startswith("tok_emb") or name.startswith("lm_head")
        if p.ndim >= 2 and not is_embed_or_head:
            hidden.append(p)
        else:
            other.append(p)
    return hidden, other


class MuonWithAuxAdam:
    """Muon on hidden 2D weights, AdamW on embeddings, head and 1D params.

    Exposes a combined ``param_groups`` list where every group carries a
    ``base_lr``, so the training loop can apply one shared schedule multiplier
    to both optimizers. Implements the pieces train.py needs: ``step``,
    ``zero_grad``, ``state_dict`` and ``load_state_dict``.
    """

    def __init__(
        self,
        hidden_params,
        other_params,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        muon_weight_decay: float = 0.0,
        ns_steps: int = 5,
        muon_plus: str = "none",
        adam_lr: float = 6e-4,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_weight_decay: float = 0.1,
    ):
        self.muon = Muon(
            hidden_params,
            lr=muon_lr,
            momentum=muon_momentum,
            ns_steps=ns_steps,
            weight_decay=muon_weight_decay,
            muon_plus=muon_plus,
        )
        self.adam = AdamW(
            other_params, lr=adam_lr, betas=adam_betas, weight_decay=adam_weight_decay
        )
        for group in self.muon.param_groups:
            group["base_lr"] = muon_lr
        for group in self.adam.param_groups:
            group["base_lr"] = adam_lr
        self.param_groups = self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.muon.step()
        self.adam.step()

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])
