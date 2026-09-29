"""KALIA model: decoder-only transformer with RMSNorm, SwiGLU and RoPE."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 10
    n_head: int = 8
    n_embd: int = 512
    context_len: int = 1024
    dropout: float = 0.0
    qk_norm: bool = False
    logit_softcap: float = 0.0
    n_kv_head: int | None = None
    n_loops: int = 1
    nope_interval: int = 0
    """Drop RoPE on every Nth layer (0 = keep RoPE everywhere). NoPE, per SmolLM3."""

    gated_residual: bool = False
    """Use the 4-branch Gated Residual stream (Qwen3.8)."""

    branch_norm: bool = False
    """4-branch stream normalisation with no gate (X19: isolates X18's -0.0436)."""


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x


def rope_tables(head_dim: int, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary position embeddings (RoPE)."""
    half = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent (even, odd) pairs in x: (B, H, T, D)."""
    t = x.size(-2)
    cos = cos[:t].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
    sin = sin[:t].unsqueeze(0).unsqueeze(0)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    """Gated feed-forward network: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = int(8 * config.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention.

    ``use_rope=False`` implements NoPE: rotary position embeddings are skipped
    entirely for that layer, relying on causal masking alone for order. SmolLM3
    validated a hybrid where every 4th layer drops RoPE -- long-context quality
    improved with no short-context cost (Yang et al. 2025, "RoPE to NoRoPE and
    Back Again"), and unlike most frontier changes it removes parameters' worth of
    machinery rather than adding any.
    """

    def __init__(self, config: GPTConfig, use_rope: bool = True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        assert config.n_head % self.n_kv_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_rope = use_rope
        self.q_norm = RMSNorm(self.head_dim) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if config.qk_norm else None
        qkv_dim = (config.n_head + 2 * self.n_kv_head) * self.head_dim
        self.qkv = nn.Linear(config.n_embd, qkv_dim, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        cos, sin = rope_tables(self.head_dim, config.context_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(
            [
                self.n_head * self.head_dim,
                self.n_kv_head * self.head_dim,
                self.n_kv_head * self.head_dim,
            ],
            dim=2,
        )
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, self.rope_cos, self.rope_sin) if self.use_rope else q
        k = apply_rope(k, self.rope_cos, self.rope_sin) if self.use_rope else k
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        # With a document mask the mask already encodes causality, so is_causal
        # must be off; without one, keep the fused causal fast path.
        if attn_mask is None:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
            )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class BranchNorm(nn.Module):
    """The gate-free half of Gated Residual: branch-wise normalisation only.

    X18's ``micro-gated`` arm finished 0.0436 nats ahead of control -- 4.4x the
    registered bar -- and the gate was then measured as non-responsive to its
    input (G-0: std across inputs 5e-06 against a mean of 0.019). So the gate is
    inert and the *structure* is the only candidate left. This class is that
    structure with the gate removed, which is what X19 needs to attribute the
    gain.

    Four quarters of the residual stream are RMSNorm'd independently and
    concatenated back to full width before the sublayers. Because RMSNorm is
    scale-invariant, ``GatedResidual`` at a near-closed gate reduces to exactly
    this multiplied by ~0.019, so the two arms differ only by the gate's
    ``w1``/``w2``.

    Not free, and the pre-registration was corrected for saying so: each
    ``RMSNorm`` carries a learnable weight, so this adds ``4 * branch_dim ==
    n_embd`` parameters per block. See Amendment 1 to X19.
    """

    def __init__(self, dim: int, n_branch: int = 4):
        super().__init__()
        assert dim % n_branch == 0, "n_embd must divide evenly into branches"
        self.n_branch = n_branch
        self.branch_dim = dim // n_branch
        self.norms = nn.ModuleList([RMSNorm(self.branch_dim) for _ in range(n_branch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise each branch independently, then concatenate to full width.

        With RMSNorm's affine weight initialised to ones this is exactly
        "renormalise each quarter of the residual stream", which is the
        operation the gated arm was performing underneath its dead gate.
        """
        parts = []
        for i, norm in enumerate(self.norms):
            sl = slice(i * self.branch_dim, (i + 1) * self.branch_dim)
            parts.append(norm(x[..., sl]))
        return torch.cat(parts, dim=-1)


class GatedResidual(nn.Module):
    """Four-branch residual stream read through a learned elementwise gate.

    Qwen3.8-Flash-Next's Gated Residual: normalise each branch independently,
    predict an elementwise gate per branch and channel from all branches, and
    average the gated branches. Widening the stream alone was worth +1.58 points
    of average accuracy; making the read data-dependent added a further +1.98 --
    while the *loss* gap was only 0.002, and in a separate case loss fell
    monotonically while accuracy saturated.

    That is the direct evidence for our own D5 (screen on benchmarks, not loss
    alone), and this is the change that would demonstrate it at 58M.
    """

    def __init__(self, dim: int, n_branch: int = 4, gate_rank: int = 32):
        super().__init__()
        assert dim % n_branch == 0, "n_embd must divide evenly into branches"
        self.n_branch = n_branch
        self.branch_dim = dim // n_branch
        self.norms = nn.ModuleList([RMSNorm(self.branch_dim) for _ in range(n_branch)])
        self.w1 = nn.Linear(dim, gate_rank, bias=True)
        self.w2 = nn.Linear(gate_rank, dim, bias=True)
        # Start almost closed so the block begins as a faithful copy of the
        # pre-norm residual path, and has to learn to open the wider stream.
        nn.init.zeros_(self.w2.weight)
        nn.init.constant_(self.w2.bias, -4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a full-width transform of the residual stream.

        Branches are normalised independently, gated elementwise, then
        *concatenated* back to full width as the block input; attention and the
        MLP still run at full width, so this adds a gated residual path without
        splitting the sublayers.

        Scope note: Qwen3.8 widens the stream to `n_branch` full-width branches
        (a real capacity increase, +1.58 accuracy) and additionally makes the read
        data-dependent (+1.98). This implements only the gating half on the
        existing width, so it tests the *mechanism*, not the widening. The
        ablation is registered as such.
        """
        gated = []
        for i, n in enumerate(self.norms):
            sl = slice(i * self.branch_dim, (i + 1) * self.branch_dim)
            b = n(x[..., sl])
            g = torch.sigmoid(self.w2(F.silu(self.w1(x)))[..., sl])
            gated.append(b * g)
        return torch.cat(gated, dim=-1)


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_index: int = 0):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        # NoPE: every nope_interval-th layer drops rotary embeddings.
        use_rope = not (
            config.nope_interval and (layer_index + 1) % config.nope_interval == 0
        )
        self.attn = CausalSelfAttention(config, use_rope=use_rope)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)
        self.gated = GatedResidual(config.n_embd) if config.gated_residual else None
        # X19: the same branch structure with the gate removed. Mutually
        # exclusive with gated_residual so an arm cannot accidentally carry both.
        assert not (config.gated_residual and config.branch_norm), (
            "branch_norm and gated_residual are alternative arms, not combinable"
        )
        self.branch = BranchNorm(config.n_embd) if config.branch_norm else None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Gated Residual transforms the stream *before* the pre-norm, as in
        # Qwen3.8: widen and gate the residual path, then run the sublayers at
        # full width on the merged result.
        pre = x if self.gated is None and self.branch is None else (
            self.gated(x) if self.gated is not None else self.branch(x)
        )
        x = x + self.attn(self.norm1(pre), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.cfg = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [Block(config, layer_index=i) for i in range(config.n_layer)]
        )
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)
        # Scaled init for residual output projections (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ):
        x = self.tok_emb(idx)
        for _ in range(self.cfg.n_loops):
            for block in self.blocks:
                x = block(x, attn_mask)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if self.cfg.logit_softcap > 0:
            cap = self.cfg.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if targets is not None:
            if loss_mask is None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                # Drop positions whose target opens a new document, so the model
                # is not trained to predict across a document boundary.
                per_token = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction="none",
                ).view_as(targets)
                kept = per_token[loss_mask]
                loss = kept.mean() if kept.numel() else per_token.mean()
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 200,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                top = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < top, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        if was_training:
            self.train()
        return idx
