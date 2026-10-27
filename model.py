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

    reverse_attention: bool = False
    """WITHDRAWN arm (X27-am1): the registered reverse-stream design attended
    to future positions, so it could read the token being predicted. Enabling
    it raises; the flag is kept as a tombstone so the lesson lives where
    someone would try to switch it on. See
    docs/preregistrations/2026-10-09-X27-amendment-1.md.
    """

    branch_norm: bool = False
    """4-branch stream normalisation with no gate (X19: isolates X18's -0.0436)."""

    static_gate: bool = False
    """Branch norm with a learned per-channel gate that never reads its input (X20)."""

    memory_slots: int = 0
    """X36: self-supervised entity workspace. 0 = off."""

    memory_chunk: int = 64
    """Tokens per workspace write. The workspace is updated at chunk ends."""

    memory_copy: bool = False
    """X36: copy head over the input prefix, gated per token."""

    memory_control: bool = False
    """X36: parameter-matched output adapter control (no memory, no copy)."""

    memory_control_hidden: int = 1024
    """Hidden width of the control adapter; chosen to match the memory params."""

    memory_gate_init: float = -4.0
    """Write-gate bias. -4 keeps the workspace closed at initialisation."""

    copy_gate_init: float = -4.0
    """Copy-gate bias. -4 keeps the copy head closed at initialisation; it opens
    only if copying reduces the language loss."""

    recurrence_weight: float = 0.1
    """Weight of the self-supervised recurrence loss during training."""

    recurrence_window: int = 64
    """A token is labelled recurring if it appears again within this many tokens."""

    recurrence_min_id: int = 1000
    """Skip the most frequent token ids in the recurrence loss (content mask)."""

    sutra: bool = False
    """X37: replace each block MLP with the SUTRA rule grammar."""

    sutra_rules: int = 64
    """Number of reusable low-rank rules (sutras) per block."""

    sutra_rank: int = 8
    """Rank of each rule transformation."""

    sutra_top_k: int = 1
    """Rules selected per token. The first implementation is top-1."""

    sutra_general_hidden: int = 1024
    """Hidden width of the always-on general path."""

    sutra_chunk: int = 64
    """Tokens per adhikara scope / anuvritti carry update."""

    sutra_carry_dim: int = 64
    """Width of the anuvritti carry state."""

    sutra_carry: bool = True
    """Enable the anuvritti carry."""

    sutra_scope: bool = True
    """Enable the adhikara scope."""

    sutra_exception: bool = False
    """Enable the utsarga/apavada exception path."""

    sutra_balance: float = 0.01
    """Weight of the Switch-style load-balancing loss."""

    sutra_exc_sparsity: float = 0.01
    """Weight of the exception-sparsity penalty."""

    sutra_control: bool = False
    """X37 control: a parameter-matched wider SwiGLU, no rules."""

    sutra_control_hidden: int = 1408
    """Hidden width that matches the SUTRA block's parameter count."""


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

    def __init__(self, config: GPTConfig, hidden: int | None = None):
        super().__init__()
        hidden = int(8 * config.n_embd / 3) if hidden is None else int(hidden)
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

    ``static_gate`` adds X20's arm: the same learned per-channel modulation, but
    with the input path removed -- ``w1``'s weight is replaced by a learned
    constant, so the gate is fixed at every position. It is the controlled test of
    whether the gain needs the gate to *read* the input at all.
    """

    def __init__(
        self,
        dim: int,
        n_branch: int = 4,
        gate_rank: int = 32,
        static_gate: bool = False,
    ):
        super().__init__()
        assert dim % n_branch == 0, "n_embd must divide evenly into branches"
        self.n_branch = n_branch
        self.branch_dim = dim // n_branch
        self.norms = nn.ModuleList([RMSNorm(self.branch_dim) for _ in range(n_branch)])
        self.static_gate = static_gate
        if static_gate:
            # A learned constant in place of w1's weight: same rank, same w2, same
            # -4.0 start, and no dependence on x anywhere. Strictly smaller than
            # GatedResidual by (dim - 1) * rank per block.
            self.gate_in = nn.Parameter(torch.zeros(gate_rank))
            self.gate_out = nn.Linear(gate_rank, dim, bias=True)
            with torch.no_grad():
                self.gate_out.weight.normal_(0.0, 0.02)
                self.gate_out.bias.fill_(-4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise each branch independently, then concatenate to full width.

        With RMSNorm's affine weight initialised to ones this is exactly
        "renormalise each quarter of the residual stream", which is the operation
        the gated arm was performing underneath its dead gate.
        """
        g = None
        if self.static_gate:
            g = torch.sigmoid(self.gate_out(F.silu(self.gate_in)))
        parts = []
        for i, norm in enumerate(self.norms):
            sl = slice(i * self.branch_dim, (i + 1) * self.branch_dim)
            b = norm(x[..., sl])
            if g is not None:
                b = b * g[..., sl]
            parts.append(b)
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


class EntityWorkspace(nn.Module):
    """X36: a small set of entity slots written from the trunk's final states.

    The workspace is updated at chunk boundaries. A write gate sees the chunk's
    mean hidden state, the mean recurrence probability, and the mean surprise
    (the model's own NLL on the chunk), and decides per slot whether to accept
    the new content. The read is a cross-attention from the current chunk to the
    slot values, with a zero-initialised output projection, so the model starts
    as its control and has to learn to use the memory.

    Slots are per sequence; the keys and the initial values are shared learned
    parameters. The workspace is a bottleneck (32 slots by default), not a
    second KV cache: the model must abstract what is worth keeping.
    """

    def __init__(self, dim: int, slots: int, gate_init: float = -4.0):
        super().__init__()
        self.slots = slots
        self.slot_keys = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.slot_values_init = nn.Parameter(torch.zeros(slots, dim))
        self.read_q = nn.Linear(dim, dim, bias=False)
        self.read_o = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.read_o.weight)
        self.write_gate = nn.Sequential(
            nn.Linear(dim + 2, dim),
            nn.SiLU(),
            nn.Linear(dim, slots),
        )
        nn.init.constant_(self.write_gate[-1].bias, gate_init)

    def initial(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.slot_values_init.unsqueeze(0).expand(batch, -1, -1).to(device)

    def read(self, h: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        q = self.read_q(h)
        scores = q @ self.slot_keys.t() / math.sqrt(h.size(-1))
        attn = F.softmax(scores, dim=-1)
        return self.read_o(attn @ values)

    def write(
        self,
        h: torch.Tensor,
        recurrence: torch.Tensor,
        surprise: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        pooled = torch.cat(
            [h.mean(dim=1), recurrence.mean(dim=1, keepdim=True),
             surprise.mean(dim=1, keepdim=True)],
            dim=-1,
        )
        gate = torch.sigmoid(self.write_gate(pooled))  # B, slots
        scores = h @ self.slot_keys.t() / math.sqrt(h.size(-1))
        assign = F.softmax(scores, dim=-1)  # B, T, slots
        mass = assign.sum(dim=1).clamp_min(1e-6)  # B, slots
        new = torch.einsum("btm,btd->bmd", assign, h) / mass.unsqueeze(-1)
        return gate.unsqueeze(-1) * new + (1.0 - gate.unsqueeze(-1)) * values


class OutputAdapter(nn.Module):
    """X36 control: a parameter-matched residual MLP at the output.

    The memory arm adds a workspace read, a copy head and a recurrence head.
    This adapter gives the control the same number of trainable parameters and a
    comparable per-token compute path, without any memory or copy mechanism. It
    is zero-initialised, so both arms start at the same function.
    """

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.down(F.silu(self.up(h)))


class CopyHead(nn.Module):
    """X36: copy distribution over the input prefix, gated per token.

    The copy logits are a causal attention from the current hidden state to the
    input-prefix hidden states. They are scattered onto the vocabulary ids of
    the prefix tokens and added to the LM logits through a learned gate. A name
    the model has already seen can be copied instead of regenerated.
    """

    def __init__(self, dim: int, vocab: int, gate_init: float = -4.0):
        super().__init__()
        self.vocab = vocab
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, 1)
        nn.init.constant_(self.gate.bias, gate_init)

    def distribution(
        self,
        h_q: torch.Tensor,
        h_k: torch.Tensor,
        idx_k: torch.Tensor,
        q_start: int,
    ) -> torch.Tensor:
        """Copy mass over the vocabulary, scattered from the causal prefix."""
        b, tq, d = h_q.shape
        tk = h_k.size(1)
        q = self.q(h_q)
        k = self.k(h_k)
        scores = q @ k.transpose(1, 2) / math.sqrt(d)
        q_pos = torch.arange(q_start, q_start + tq, device=h_q.device).unsqueeze(1)
        k_pos = torch.arange(tk, device=h_q.device).unsqueeze(0)
        allowed = k_pos <= q_pos
        scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        pos = F.softmax(scores, dim=-1)
        copy = torch.zeros(b, tq, self.vocab, device=h_q.device)
        copy.scatter_add_(
            2,
            idx_k.unsqueeze(1).expand(-1, tq, -1),
            pos,
        )
        return copy

    def forward(
        self,
        h_q: torch.Tensor,
        logits_q: torch.Tensor,
        h_k: torch.Tensor,
        idx_k: torch.Tensor,
        q_start: int,
    ) -> torch.Tensor:
        copy = self.distribution(h_q, h_k, idx_k, q_start)
        gate = torch.sigmoid(self.gate(h_q))
        return logits_q + gate * torch.log(copy + 1e-6)


class RecurrenceHead(nn.Module):
    """X36: predicts whether a token/name will appear again in the next window.

    The label is self-supervised: a token is positive if the same id appears in
    the next `recurrence_window` tokens. The head is trained jointly with the LM
    and its output gates the workspace write, so the model learns what to keep.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h).squeeze(-1)


class SutraMLP(nn.Module):
    """X37: a Pāṇinian rule grammar in place of the block MLP.

    The always-on general path is a small SwiGLU. A router selects one of R
    low-rank rules per token; the rule output is added to the general output.
    The router sees the token, an **anuvritti** carry from the previous chunk and
    an **adhikara** scope vector (the previous chunk's mean). A Switch-style
    load-balancing loss keeps the codebook alive. An optional exception path
    (utsarga/apavada) can override the general rule when a conflict detector
    fires.

    The carry and scope are updated only at chunk ends and only for the *next*
    chunk, so a token can never see a future token through the grammar.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.sutra_top_k == 1, "the first SUTRA implementation is top-1"
        self.dim = config.n_embd
        self.rules = config.sutra_rules
        self.rank = config.sutra_rank
        self.chunk = config.sutra_chunk
        self.carry_dim = config.sutra_carry_dim
        self.carry_on = config.sutra_carry
        self.scope_on = config.sutra_scope
        self.exception = config.sutra_exception
        self.balance = config.sutra_balance
        self.exc_sparsity = config.sutra_exc_sparsity
        self.general = SwiGLU(config, hidden=config.sutra_general_hidden)
        self.rules_A = nn.Parameter(torch.randn(self.rules, self.rank, self.dim) * 0.02)
        self.rules_B = nn.Parameter(torch.randn(self.rules, self.dim, self.rank) * 0.02)
        self.rule_emb = nn.Parameter(torch.randn(self.rules, self.carry_dim) * 0.02)
        self.router = nn.Linear(2 * self.dim + self.carry_dim, self.rules)
        self.carry_gate = nn.Linear(self.dim + self.carry_dim, 1)
        if self.exception:
            self.exc_A = nn.Parameter(torch.randn(4, self.dim) * 0.02)
            self.exc_B = nn.Parameter(torch.randn(self.dim, 4) * 0.02)
            self.exc_gate = nn.Linear(2 * self.dim + self.carry_dim, 1)
            nn.init.constant_(self.exc_gate.bias, -4.0)
        self.aux_loss = None
        self.last_usage = None
        self.last_probs = None
        self.last_exc = None

    def _rule(self, x: torch.Tensor, topi: torch.Tensor) -> torch.Tensor:
        A = self.rules_A[topi]  # B, T, rank, dim
        B = self.rules_B[topi]  # B, T, dim, rank
        z = torch.einsum("btrd,btd->btr", A, x)
        return torch.einsum("btdr,btr->btd", B, z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        out = x.new_zeros(b, t, d)
        carry = x.new_zeros(b, self.carry_dim)
        scope = x.new_zeros(b, d)
        usage = x.new_zeros(self.rules, device=x.device)
        prob_sum = x.new_zeros(self.rules, device=x.device)
        aux = x.new_zeros((), device=x.device)
        exc_sum = x.new_zeros((), device=x.device)
        n_chunks = 0
        for start in range(0, t, self.chunk):
            end = min(start + self.chunk, t)
            xc = x[:, start:end]
            n = end - start
            ctx = torch.cat(
                [
                    xc,
                    carry.unsqueeze(1).expand(-1, n, -1),
                    scope.unsqueeze(1).expand(-1, n, -1),
                ],
                dim=-1,
            )
            probs = F.softmax(self.router(ctx), dim=-1)
            topv, topi = probs.max(dim=-1)
            y = self.general(xc) + self._rule(xc, topi) * topv.unsqueeze(-1)
            if self.exception:
                c = torch.sigmoid(self.exc_gate(ctx))
                e = torch.einsum("btd,rd->btr", xc, self.exc_A)
                e = torch.einsum("btr,dr->btd", e, self.exc_B)
                y = y + c * e
                aux = aux + self.exc_sparsity * c.mean()
                exc_sum = exc_sum + c.mean()
            out[:, start:end] = y
            f = F.one_hot(topi, self.rules).float().mean(dim=(0, 1))
            p = probs.mean(dim=(0, 1))
            aux = aux + self.balance * self.rules * (f * p).sum()
            usage = usage + f
            prob_sum = prob_sum + p
            n_chunks += 1
            if self.carry_on:
                gate = torch.sigmoid(self.carry_gate(torch.cat([scope, carry], dim=-1)))
                carry = gate * carry + (1.0 - gate) * (probs.mean(dim=1) @ self.rule_emb)
            if self.scope_on:
                scope = xc.mean(dim=1)
        self.aux_loss = aux
        self.last_usage = (usage / max(1, n_chunks)).detach()
        self.last_probs = (prob_sum / max(1, n_chunks)).detach()
        self.last_exc = (exc_sum / max(1, n_chunks)).detach() if self.exception else None
        return out


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_index: int = 0):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        # NoPE: every nope_interval-th layer drops rotary embeddings.
        use_rope = not (
            config.nope_interval and (layer_index + 1) % config.nope_interval == 0
        )
        # X27 was withdrawn before any GPU hour: as designed, the reverse
        # stream attended to future positions and could read the token being
        # predicted. The flag is a tripwire, not a usable arm.
        if config.reverse_attention:
            raise ValueError(
                "RCAA is withdrawn (X27-am1): the registered design attended to "
                "future positions, so the model could read the token it was "
                "asked to predict. See "
                "docs/preregistrations/2026-10-09-X27-amendment-1.md"
            )
        self.attn = CausalSelfAttention(config, use_rope=use_rope)
        self.norm2 = RMSNorm(config.n_embd)
        if config.sutra:
            self.mlp = SutraMLP(config)
        elif config.sutra_control:
            self.mlp = SwiGLU(config, hidden=config.sutra_control_hidden)
        else:
            self.mlp = SwiGLU(config)
        self.gated = GatedResidual(config.n_embd) if config.gated_residual else None
        # X19: the same branch structure with the gate removed. Mutually
        # exclusive with gated_residual so an arm cannot accidentally carry both.
        assert not (config.gated_residual and config.branch_norm), (
            "branch_norm and gated_residual are alternative arms, not combinable"
        )
        # A static_gate without branch_norm would construct a plain control model
        # while the config claims the X20 arm. Silent no-op configs are how a
        # clean, confident, wrong number happens.
        assert not (config.static_gate and not config.branch_norm), (
            "static_gate requires branch_norm; without it the model is a plain "
            "control and the config would be lying about which arm is running"
        )
        self.branch = (
            BranchNorm(config.n_embd, static_gate=config.static_gate)
            if config.branch_norm
            else None
        )

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
        assert not (config.sutra and config.sutra_control), (
            "sutra and sutra_control are separate arms"
        )
        assert not ((config.sutra or config.sutra_control)
                    and (config.memory_slots > 0 or config.memory_copy
                         or config.memory_control)), (
            "SUTRA and the X36 memory are separate arms; do not combine them in "
            "one experiment"
        )
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [Block(config, layer_index=i) for i in range(config.n_layer)]
        )
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # weight tying
        # Initialise the base first, before any optional module is constructed.
        # The base weights must be identical across the control/treatment arms;
        # constructing the extra modules before self.apply would shift the RNG
        # stream and silently give the arms different starting models.
        self.apply(self._init_weights)
        # Scaled init for residual output projections (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        # X36: output-level entity memory, copy head and control adapter. They
        # are built after the base initialisation; their own init is applied
        # only to them, then the zeros/gates are re-asserted (the global re-init
        # is exactly what silently killed the X18 gate).
        if config.memory_control:
            assert config.memory_slots == 0 and not config.memory_copy, (
                "memory_control is the no-memory control; it cannot also carry "
                "the workspace or the copy head"
            )
            self.adapter = OutputAdapter(config.n_embd, config.memory_control_hidden)
        else:
            self.adapter = None
        self.memory = (
            EntityWorkspace(config.n_embd, config.memory_slots, config.memory_gate_init)
            if config.memory_slots > 0
            else None
        )
        self.recurrence = RecurrenceHead(config.n_embd) if self.memory is not None else None
        self.copy = (
            CopyHead(config.n_embd, config.vocab_size, config.copy_gate_init)
            if config.memory_copy
            else None
        )
        for module in (self.adapter, self.memory, self.recurrence, self.copy):
            if module is not None:
                module.apply(self._init_weights)
        if self.adapter is not None:
            nn.init.zeros_(self.adapter.down.weight)
        if self.memory is not None:
            nn.init.zeros_(self.memory.read_o.weight)
            nn.init.constant_(self.memory.write_gate[-1].bias, config.memory_gate_init)
        if self.copy is not None:
            nn.init.constant_(self.copy.gate.bias, config.copy_gate_init)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _recurrence_labels(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Self-supervised recurrence labels and the content-token mask.

        A token is positive if the same id appears again within
        ``recurrence_window`` positions. The most frequent token ids are
        excluded from the loss (the content mask); the head still runs on every
        token, because the write gate uses its output everywhere.
        """
        b, t = idx.shape
        labels = torch.zeros_like(idx, dtype=torch.bool)
        for d in range(1, self.cfg.recurrence_window + 1):
            if d >= t:
                break
            labels[:, : t - d] |= idx[:, : t - d] == idx[:, d:]
        mask = (idx > self.cfg.recurrence_min_id) & (idx != 50256)
        return labels, mask

    def _finish(self, logits, targets, loss_mask):
        if self.cfg.logit_softcap > 0:
            cap = self.cfg.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if targets is not None:
            if loss_mask is None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                per_token = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction="none",
                ).view_as(targets)
                kept = per_token[loss_mask]
                loss = kept.mean() if kept.numel() else per_token.mean()
        return logits, loss

    def _forward_memory(self, h, idx, targets, loss_mask):
        """X36: chunked workspace read/write, copy head and recurrence loss.

        The trunk has already run causally over the full sequence. The workspace
        is updated at chunk ends; chunk c reads the values written by chunks
        < c, so no token can influence an earlier token. The output read has a
        zero-initialised projection; the model starts as its control.
        """
        b, t, _ = h.shape
        values = self.memory.initial(b, h.device)
        chunk = self.cfg.memory_chunk
        rec_labels, rec_mask = self._recurrence_labels(idx)
        logits_chunks = []
        lm_losses = []
        rec_losses = []
        for start in range(0, t, chunk):
            end = min(start + chunk, t)
            hc = h[:, start:end]
            hc = hc + self.memory.read(hc, values)
            logits_c = self.lm_head(hc)
            if self.copy is not None:
                logits_c = self.copy(hc, logits_c, h[:, :end], idx[:, :end], start)
            if self.cfg.logit_softcap > 0:
                cap = self.cfg.logit_softcap
                logits_c = cap * torch.tanh(logits_c / cap)
            logits_chunks.append(logits_c)
            r_logits = self.recurrence(hc)
            r = torch.sigmoid(r_logits)
            if targets is not None:
                per_token = F.cross_entropy(
                    logits_c.view(-1, logits_c.size(-1)),
                    targets[:, start:end].reshape(-1),
                    reduction="none",
                ).view(b, end - start)
                if loss_mask is None:
                    lm_losses.append(per_token.mean())
                else:
                    keep = loss_mask[:, start:end]
                    lm_losses.append(per_token[keep].mean() if keep.any() else per_token.mean())
                surprise = per_token.detach()
            else:
                surprise = torch.zeros(b, end - start, device=h.device)
            values = self.memory.write(hc, r, surprise, values)
            if targets is not None and self.training and self.cfg.recurrence_weight > 0:
                keep = rec_mask[:, start:end]
                if keep.any():
                    rec_losses.append(
                        F.binary_cross_entropy_with_logits(
                            r_logits[keep],
                            rec_labels[:, start:end][keep].float(),
                        )
                    )
        logits = torch.cat(logits_chunks, dim=1)
        if targets is None:
            return logits, None
        loss = torch.stack(lm_losses).mean()
        if rec_losses:
            loss = loss + self.cfg.recurrence_weight * torch.stack(rec_losses).mean()
        return logits, loss

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
        h = self.norm_f(x)
        if self.adapter is not None:
            h = self.adapter(h)
        if self.memory is not None:
            return self._forward_memory(h, idx, targets, loss_mask)
        logits = self.lm_head(h)
        if self.copy is not None:
            logits = self.copy(h, logits, h, idx, 0)
        logits, loss = self._finish(logits, targets, loss_mask)
        if self.cfg.sutra and self.training and loss is not None:
            aux = torch.zeros((), device=idx.device)
            for block in self.blocks:
                if getattr(block.mlp, "aux_loss", None) is not None:
                    aux = aux + block.mlp.aux_loss
            loss = loss + aux
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 200,
        generator: torch.Generator | None = None,
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
            sample = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat((idx, sample), dim=1)
        if was_training:
            self.train()
        return idx
