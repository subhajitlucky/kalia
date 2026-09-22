# KALIA Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use the executing-plans skill to implement this plan task-by-task (or subagent-driven-development for task-by-task dispatch).

**Goal:** Build and train KALIA — a from-scratch ~58M-parameter GPT-style language model — on free Kaggle T4×2 GPUs, with a resumable training pipeline that checkpoints to HuggingFace Hub.

**Architecture:** Decoder-only transformer (RMSNorm, SwiGLU, RoPE, weight tying), GPT-2 BPE tokenizer, fp16 AMP, AdamW with cosine schedule. Data: TinyStories + FineWeb-Edu tokenized to uint16 `.bin` shards. Training runs in finite Kaggle sessions, saving/resuming checkpoints via HF Hub.

**Tech Stack:** Python 3.12, PyTorch (CPU locally, CUDA on Kaggle), tiktoken, numpy, pyyaml, datasets, huggingface_hub, pytest.

**Design doc:** `docs/plans/2026-09-22-kalia-design.md`

---

## Conventions for every task

- Local dev runs from the repo root.
- Run tests with: `python -m pytest tests/ -v`
- Commit after every task with the message shown.
- Never commit: `*.bin`, `*.pt`, `data/`, `out/`, `.venv/` (already in `.gitignore`).

---

### Task 1: Repo scaffold

**Files:**
- Create: `.gitignore`, `requirements.txt`, `LICENSE`, `configs/kalia-m.yaml`, `configs/smoke.yaml`, `README.md` (placeholder)

**Step 1: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
data/
out/
*.bin
*.pt
.pytest_cache/
.kaggle/
```

**Step 2: Create `requirements.txt`**

```
torch>=2.4
tiktoken>=0.7
numpy>=1.26
pyyaml>=6.0
datasets>=3.0
huggingface_hub>=0.23
pytest>=8.0
```

**Step 3: Create `LICENSE`**

MIT License, copyright 2026 KALIA contributors.

**Step 4: Create `configs/kalia-m.yaml`**

```yaml
model:
  vocab_size: 50257
  n_layer: 10
  n_head: 8
  n_embd: 512
  context_len: 1024
  dropout: 0.0
train:
  micro_batch_size: 32
  grad_accum_steps: 8
  max_steps: 4770
  learning_rate: 6.0e-4
  min_lr_ratio: 0.1
  warmup_steps: 500
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  eval_interval: 250
  eval_steps: 50
  sample_interval: 500
  sample_tokens: 80
  checkpoint_interval_minutes: 30
  log_interval: 10
  seed: 1337
```

**Step 5: Create `configs/smoke.yaml` (tiny CPU test config)**

```yaml
model:
  vocab_size: 256
  n_layer: 2
  n_head: 2
  n_embd: 64
  context_len: 64
  dropout: 0.0
train:
  micro_batch_size: 4
  grad_accum_steps: 2
  max_steps: 10
  learning_rate: 1.0e-3
  min_lr_ratio: 0.1
  warmup_steps: 2
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  eval_interval: 5
  eval_steps: 2
  sample_interval: 0
  sample_tokens: 16
  checkpoint_interval_minutes: 0
  log_interval: 1
  seed: 1337
```

**Step 6: Create placeholder `README.md`**

```markdown
# KALIA

A from-scratch ~58M-parameter language model, trained on free Kaggle GPUs.
Full documentation coming in Task 13.
```

**Step 7: Commit**

```bash
git add .gitignore requirements.txt LICENSE configs/ README.md
git commit -m "chore: repo scaffold, KALIA-M + smoke configs"
```

---

### Task 2: `model.py` — RMSNorm

**Files:**
- Create: `model.py`
- Create: `tests/test_model.py`

**Step 1: Write the failing test**

```python
# tests/test_model.py
import torch
from model import RMSNorm


def test_rmsnorm_shape_and_scale():
    norm = RMSNorm(16)
    x = torch.randn(2, 5, 16)
    out = norm(x)
    assert out.shape == x.shape
    # With weight=1, each vector's RMS becomes 1
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rmsnorm_learned_weight_applies():
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    x = torch.randn(3, 8)
    out = norm(x)
    base = RMSNorm(8)(x)
    assert torch.allclose(out, base * 2.0, atol=1e-5)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model'`

**Step 3: Write minimal implementation**

```python
# model.py
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


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat(model): RMSNorm"
```

---

### Task 3: `model.py` — RoPE

**Files:**
- Modify: `model.py` (append)
- Modify: `tests/test_model.py` (append)

**Step 1: Write the failing test**

```python
# tests/test_model.py (append)
from model import rope_tables, apply_rope


def test_rope_preserves_vector_norms():
    cos, sin = rope_tables(head_dim=8, max_seq_len=16)
    x = torch.randn(2, 4, 16, 8)  # (B, H, T, D)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_position_zero_is_identity():
    cos, sin = rope_tables(head_dim=8, max_seq_len=16)
    x = torch.randn(1, 2, 16, 8)
    out = apply_rope(x, cos, sin)
    # Position 0 has angle 0 -> no rotation
    assert torch.allclose(out[:, :, 0], x[:, :, 0], atol=1e-6)


def test_rope_uses_only_past_positions_shape():
    cos, sin = rope_tables(head_dim=8, max_seq_len=16)
    x = torch.randn(1, 1, 4, 8)  # short sequence
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL — `cannot import name 'rope_tables'`

**Step 3: Write minimal implementation**

```python
# model.py (append)

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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat(model): RoPE with rotation-preserving tests"
```

---

### Task 4: `model.py` — SwiGLU

**Files:**
- Modify: `model.py` (append)
- Modify: `tests/test_model.py` (append)

**Step 1: Write the failing test**

```python
# tests/test_model.py (append)
from model import SwiGLU, GPTConfig


def test_swiglu_shape():
    cfg = GPTConfig(n_embd=64)
    mlp = SwiGLU(cfg)
    x = torch.randn(2, 5, 64)
    assert mlp(x).shape == x.shape
    hidden = mlp.gate.out_features
    assert hidden % 64 == 0  # hidden dim rounded to multiple of 64
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL — `cannot import name 'SwiGLU'`

**Step 3: Write minimal implementation**

```python
# model.py (append)

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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: 6 passed

**Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat(model): SwiGLU feed-forward"
```

---

### Task 5: `model.py` — Causal self-attention

**Files:**
- Modify: `model.py` (append)
- Modify: `tests/test_model.py` (append)

**Step 1: Write the failing test**

```python
# tests/test_model.py (append)
from model import CausalSelfAttention


def test_attention_shape():
    cfg = GPTConfig(n_embd=64, n_head=4, context_len=32)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 16, 64)
    assert attn(x).shape == x.shape


def test_attention_is_causal():
    torch.manual_seed(0)
    cfg = GPTConfig(n_embd=64, n_head=4, context_len=32)
    attn = CausalSelfAttention(cfg)
    attn.eval()
    x = torch.randn(1, 8, 64)
    x_future_changed = x.clone()
    x_future_changed[:, -1] += 10.0  # change last token only
    with torch.no_grad():
        out_a = attn(x)
        out_b = attn(x_future_changed)
    # Outputs for all positions before the changed token must be identical
    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-6)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL — `cannot import name 'CausalSelfAttention'`

**Step 3: Write minimal implementation**

```python
# model.py (append)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        cos, sin = rope_tables(self.head_dim, config.context_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: 8 passed

**Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat(model): causal self-attention with RoPE"
```

---

### Task 6: `model.py` — GPT assembly

**Files:**
- Modify: `model.py` (append)
- Modify: `tests/test_model.py` (append)

**Step 1: Write the failing test**

```python
# tests/test_model.py (append)
from model import GPT


def test_gpt_kalia_m_param_count():
    model = GPT(GPTConfig())  # KALIA-M default config
    n = model.num_params()
    assert 50_000_000 < n < 65_000_000, n


def test_gpt_forward_and_backward():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, context_len=32)
    model = GPT(cfg)
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 256)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_gpt_weight_tying():
    model = GPT(GPTConfig(vocab_size=256, n_layer=1, n_head=2, n_embd=64, context_len=16))
    assert model.lm_head.weight is model.tok_emb.weight


def test_gpt_generate_shape():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, context_len=16)
    model = GPT(cfg)
    prompt = torch.randint(0, 256, (1, 4))
    out = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=None)
    assert out.shape == (1, 12)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL — `cannot import name 'GPT'`

**Step 3: Write minimal implementation**

```python
# model.py (append)

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.cfg = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: 12 passed (param count prints ~57.9M)

**Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat(model): GPT assembly with weight tying and generation"
```

---

### Task 7: `data.py` — memory-mapped batches

**Files:**
- Create: `data.py`
- Create: `tests/test_data.py`

**Step 1: Write the failing test**

```python
# tests/test_data.py
import numpy as np
import torch
from data import TokenDataset


def _write_bin(path, values):
    np.array(values, dtype=np.uint16).tofile(path)


def test_batch_shapes_and_shift(tmp_path):
    values = list(range(1000))
    path = tmp_path / "train.bin"
    _write_bin(path, values)
    ds = TokenDataset(path, context_len=16)
    g = torch.Generator().manual_seed(0)
    x, y = ds.get_batch(batch_size=4, device=torch.device("cpu"), generator=g)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert torch.equal(y[:, :-1], x[:, 1:])  # y is x shifted by one


def test_deterministic_with_seed(tmp_path):
    path = tmp_path / "train.bin"
    _write_bin(path, list(range(1000)))
    ds = TokenDataset(path, context_len=16)
    x1, _ = ds.get_batch(4, torch.device("cpu"), torch.Generator().manual_seed(42))
    x2, _ = ds.get_batch(4, torch.device("cpu"), torch.Generator().manual_seed(42))
    assert torch.equal(x1, x2)


def test_rejects_short_file(tmp_path):
    path = tmp_path / "short.bin"
    _write_bin(path, [1, 2, 3])
    try:
        TokenDataset(path, context_len=16)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL — `No module named 'data'`

**Step 3: Write minimal implementation**

```python
# data.py
"""Memory-mapped uint16 token shards for KALIA training."""

from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    """Random contiguous windows from a flat uint16 token file."""

    def __init__(self, path: str | Path, context_len: int):
        self.path = Path(path)
        self.context_len = context_len
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.tokens) < context_len + 2:
            raise ValueError(
                f"{self.path} has too few tokens ({len(self.tokens)}) for context_len={context_len}"
            )

    def __len__(self) -> int:
        return len(self.tokens)

    def get_batch(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.tokens) - self.context_len - 1
        starts = torch.randint(max_start, (batch_size,), generator=generator)
        x = torch.stack(
            [torch.from_numpy(self.tokens[i : i + self.context_len].astype(np.int64)) for i in starts]
        )
        y = torch.stack(
            [
                torch.from_numpy(self.tokens[i + 1 : i + 1 + self.context_len].astype(np.int64))
                for i in starts
            ]
        )
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
```

Note: `y[:, :-1] == x[:, 1:]` holds only when windows don't overlap — but batches are random windows, so within each row, `y[:, :-1]` = tokens `i+1..i+ctx-1` and `x[:, 1:]` = tokens `i+1..i+ctx-1`. Yes, per-row equality holds regardless of overlap between rows. ✓

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add data.py tests/test_data.py
git commit -m "feat(data): memmap token dataset with shift-by-one batches"
```

---

### Task 8: `prepare.py` — tokenize sources to `.bin`

**Files:**
- Create: `prepare.py`
- Create: `tests/test_prepare.py`

**Step 1: Write the failing test**

```python
# tests/test_prepare.py
import json
import numpy as np
from prepare import tokenize_documents, SOURCES


class FakeEncoder:
    """Maps each character to its ord() value; eot = 255."""

    eot_token = 255

    def encode_ordinary(self, text):
        return [ord(ch) for ch in text]


def test_tokenize_writes_uint16_and_eot(tmp_path):
    docs = [{"text": "abc"}, {"text": "de"}]
    out = tmp_path / "train.bin"
    n = tokenize_documents(docs, FakeEncoder(), out, max_tokens=1000)
    assert n == 3 + 1 + 2 + 1  # chars + eot per doc
    arr = np.fromfile(out, dtype=np.uint16)
    assert arr.tolist() == [97, 98, 99, 255, 100, 101, 255]


def test_tokenize_respects_max_tokens(tmp_path):
    docs = [{"text": "x" * 100} for _ in range(10)]
    out = tmp_path / "train.bin"
    n = tokenize_documents(docs, FakeEncoder(), out, max_tokens=50)
    assert n >= 50  # stops at/after the limit, never writes more than one doc past it
    assert n <= 202


def test_sources_registry():
    assert set(SOURCES) == {"tinystories", "fineweb"}
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prepare.py -v`
Expected: FAIL — `No module named 'prepare'`

**Step 3: Write minimal implementation**

```python
# prepare.py
"""Tokenize text sources into uint16 .bin shards for KALIA training.

Run this on a Kaggle CPU notebook (CPU quota is unlimited) or locally for smoke tests.
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

WRITE_CHUNK = 1_000_000  # tokens buffered before flushing to disk


def iter_tinystories() -> Iterator[dict]:
    from datasets import load_dataset

    yield from load_dataset("roneneldan/TinyStories", split="train", streaming=True)


def iter_fineweb() -> Iterator[dict]:
    from datasets import load_dataset

    yield from load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True
    )


SOURCES: dict[str, Callable[[], Iterable[dict]]] = {
    "tinystories": iter_tinystories,
    "fineweb": iter_fineweb,
}


def tokenize_documents(
    docs: Iterable[dict],
    encoder,
    out_path: str | Path,
    max_tokens: int,
) -> int:
    """Tokenize documents to a uint16 file. Returns tokens written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    buffer: list[int] = []

    def flush(fh) -> None:
        nonlocal buffer, written
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(fh)
            written += len(buffer)
            buffer = []

    with open(out_path, "wb") as fh:
        for doc in docs:
            ids = encoder.encode_ordinary(doc["text"])
            ids.append(encoder.eot_token)
            buffer.extend(ids)
            if len(buffer) >= WRITE_CHUNK:
                flush(fh)
            if written + len(buffer) >= max_tokens:
                break
        flush(fh)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize a source into a .bin shard")
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--out", type=Path, required=True, help="output .bin path")
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--meta", type=Path, default=None, help="optional meta.json path")
    args = parser.parse_args()

    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    docs = SOURCES[args.source]()
    n = tokenize_documents(docs, encoder, args.out, args.max_tokens)
    print(f"wrote {n} tokens to {args.out}")

    if args.meta is not None:
        meta = {}
        if args.meta.exists():
            meta = json.loads(args.meta.read_text())
        meta[args.out.name] = {"source": args.source, "tokens": n}
        args.meta.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_prepare.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add prepare.py tests/test_prepare.py
git commit -m "feat(prepare): streaming tokenizer to uint16 shards"
```

---

### Task 9: `train.py` — training loop (single process, CPU-capable)

**Files:**
- Create: `train.py`
- Create: `tests/test_train_smoke.py`

**Step 1: Write the failing test**

```python
# tests/test_train_smoke.py
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _make_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.array(rng.integers(0, 256, 8192), dtype=np.uint16).tofile(data_dir / "train.bin")
    np.array(rng.integers(0, 256, 2048), dtype=np.uint16).tofile(data_dir / "val.bin")


def test_smoke_train_and_resume(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    _make_data(data_dir)

    base = [
        sys.executable, "train.py",
        "--config", "configs/smoke.yaml",
        "--data-dir", str(data_dir),
        "--out-dir", str(out_dir),
    ]
    r = subprocess.run(base, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert (out_dir / "ckpt.pt").exists()
    assert (out_dir / "train_log.csv").exists()

    # Resume for 10 more steps: total 20
    r = subprocess.run(base + ["--resume", "--max-steps", "20"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    import torch

    ckpt = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ckpt["step"] == 20
    assert ckpt["tokens"] == 20 * 4 * 2 * 64  # steps * micro_batch * grad_accum * context_len
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_smoke.py -v`
Expected: FAIL — `train.py` not found / exits nonzero

**Step 3: Write minimal implementation**

```python
# train.py
"""Train KALIA: single-process or DDP (torchrun), resumable, time-budgeted."""

import argparse
import csv
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW

from data import TokenDataset
from model import GPT, GPTConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    p.add_argument("--resume", action="store_true", help="resume from out_dir/ckpt.pt or --hub-repo")
    p.add_argument("--hub-repo", type=str, default=None, help="HF repo id for checkpoint sync")
    p.add_argument("--max-steps", type=int, default=None, help="override config max_steps")
    p.add_argument("--max-minutes", type=float, default=None, help="stop after this many minutes")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def setup_distributed():
    if "RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world
    return 0, 1


def get_lr(step: int, train_cfg: dict) -> float:
    peak = train_cfg["learning_rate"]
    if step < train_cfg["warmup_steps"]:
        return peak * (step + 1) / train_cfg["warmup_steps"]
    if step >= train_cfg["max_steps"]:
        return peak * train_cfg["min_lr_ratio"]
    progress = (step - train_cfg["warmup_steps"]) / max(1, train_cfg["max_steps"] - train_cfg["warmup_steps"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (train_cfg["min_lr_ratio"] + coeff * (1 - train_cfg["min_lr_ratio"]))


def save_checkpoint(path: Path, model, optimizer, step: int, tokens: int, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens": tokens,
            "config": config,
        },
        path,
    )


def push_to_hub(local_path: Path, repo_id: str, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo, repo_id=repo_id)


def pull_from_hub(repo_id: str, path_in_repo: str, local_path: Path) -> bool:
    from huggingface_hub import hf_hub_download

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id, filename=path_in_repo, token=os.environ.get("HF_TOKEN")
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "start fresh"
        print(f"resume: no checkpoint pulled ({exc})")
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(Path(downloaded).read_bytes())
    return True


@torch.no_grad()
def evaluate(model, dataset, eval_steps: int) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(eval_steps):
        x, y = dataset.get_batch(8, next(model.parameters()).device)
        _, loss = model(x, y)
        total += loss.item()
    if was_training:
        model.train()
    return total / eval_steps


@torch.no_grad()
def print_sample(model, cfg: GPTConfig, sample_tokens: int) -> None:
    raw = model.module if hasattr(model, "module") else model
    device = next(raw.parameters()).device
    prompt = torch.tensor([[464]], device=device)  # "Once" via GPT-2 BPE
    out = raw.generate(prompt, max_new_tokens=sample_tokens, temperature=0.8, top_k=200)
    print("sample ids:", out[0].tolist()[:20], "...")


def main(argv=None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    train_cfg = dict(config["train"])
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps
    if args.seed is not None:
        train_cfg["seed"] = args.seed

    rank, world = setup_distributed()
    is_master = rank == 0
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    is_cuda = device.type == "cuda"

    torch.manual_seed(train_cfg["seed"] + rank)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = GPTConfig(**config["model"])
    model = GPT(model_cfg).to(device)
    if is_master:
        print(f"KALIA params: {model.num_params():,} | device: {device} | world: {world}")

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=train_cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    start_step, tokens_seen = 0, 0
    ckpt_path = args.out_dir / "ckpt.pt"
    if args.resume:
        if args.hub_repo:
            pull_from_hub(args.hub_repo, "checkpoints/ckpt.pt", ckpt_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step, tokens_seen = ckpt["step"], ckpt["tokens"]
            if is_master:
                print(f"resumed from step {start_step} ({tokens_seen:,} tokens)")

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()]
        )

    train_ds = TokenDataset(args.data_dir / "train.bin", model_cfg.context_len)
    val_path = args.data_dir / "val.bin"
    val_ds = TokenDataset(val_path, model_cfg.context_len) if val_path.exists() else None
    generator = torch.Generator().manual_seed(train_cfg["seed"] + rank)

    log_path = args.out_dir / "train_log.csv"
    if is_master:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if not log_path.exists() or start_step == 0:
            with open(log_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["step", "loss", "lr", "tokens", "elapsed_s"])

    tokens_per_step = train_cfg["micro_batch_size"] * train_cfg["grad_accum_steps"] * model_cfg.context_len * world
    start_time = time.time()
    last_ckpt_time = start_time
    model.train()

    step = start_step
    try:
        while step < train_cfg["max_steps"]:
            if args.max_minutes is not None and (time.time() - start_time) / 60 >= args.max_minutes:
                if is_master:
                    print("time budget reached; stopping cleanly")
                break

            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            t0 = time.time()
            for _ in range(train_cfg["grad_accum_steps"]):
                x, y = train_ds.get_batch(train_cfg["micro_batch_size"], device, generator)
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=is_cuda):
                    _, loss = model(x, y)
                    loss = loss / train_cfg["grad_accum_steps"]
                scaler.scale(loss).backward()
                loss_total += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            lr = get_lr(step, train_cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()

            tokens_seen += tokens_per_step
            step += 1

            if is_master and step % train_cfg["log_interval"] == 0:
                elapsed = time.time() - t0
                tok_per_s = tokens_per_step / elapsed
                print(f"step {step}/{train_cfg['max_steps']} | loss {loss_total:.4f} | lr {lr:.2e} | {tok_per_s:,.0f} tok/s")
                with open(log_path, "a", newline="") as fh:
                    csv.writer(fh).writerow([step, f"{loss_total:.4f}", f"{lr:.6e}", tokens_seen, f"{time.time() - start_time:.1f}"])

            if val_ds is not None and train_cfg["eval_interval"] > 0 and step % train_cfg["eval_interval"] == 0:
                if is_master:
                    val_loss = evaluate(model, val_ds, train_cfg["eval_steps"])
                    print(f"step {step} | val loss {val_loss:.4f}")

            if train_cfg["sample_interval"] > 0 and step % train_cfg["sample_interval"] == 0 and is_master:
                print_sample(model, model_cfg, train_cfg["sample_tokens"])

            due = train_cfg["checkpoint_interval_minutes"] > 0 and (
                time.time() - last_ckpt_time
            ) / 60 >= train_cfg["checkpoint_interval_minutes"]
            if due and is_master:
                save_checkpoint(ckpt_path, model, optimizer, step, tokens_seen, config)
                print(f"checkpoint saved at step {step}")
                if args.hub_repo:
                    push_to_hub(ckpt_path, args.hub_repo, "checkpoints/ckpt.pt")
                    push_to_hub(log_path, args.hub_repo, "logs/train_log.csv")
                last_ckpt_time = time.time()

        if is_master:
            save_checkpoint(ckpt_path, model, optimizer, step, tokens_seen, config)
            print(f"final checkpoint saved at step {step}")
            if args.hub_repo:
                push_to_hub(ckpt_path, args.hub_repo, "checkpoints/ckpt.pt")
                push_to_hub(log_path, args.hub_repo, "logs/train_log.csv")
    finally:
        if world > 1:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train_smoke.py -v`
Expected: PASS (takes ~30-60s on CPU). If `torch.load(..., weights_only=False)` warns, that's fine.

**Step 5: Commit**

```bash
git add train.py tests/test_train_smoke.py
git commit -m "feat(train): resumable, time-budgeted training loop with HF sync"
```

---

### Task 10: `sample.py` — generate from a checkpoint

**Files:**
- Create: `sample.py`
- Create: `tests/test_sample.py`

**Step 1: Write the failing test**

```python
# tests/test_sample.py
import numpy as np
import torch
from model import GPT, GPTConfig
from sample import generate_text


class FakeEncoder:
    class _Enc:
        eot_token = 255

        def encode_ordinary(self, text):
            return [ord(c) % 256 for c in text]

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    def __new__(cls):
        return cls._Enc()


def test_generate_text_length(tmp_path):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=256, n_layer=1, n_head=2, n_embd=64, context_len=32)
    model = GPT(cfg)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "config": {"model": cfg.__dict__, "train": {}}}, ckpt)
    out = generate_text(ckpt, "ab", max_new_tokens=10, encoder=FakeEncoder())
    assert isinstance(out, str)
    assert out.startswith("ab")
    assert len(out) == 12
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sample.py -v`
Expected: FAIL — `No module named 'sample'`

**Step 3: Write minimal implementation**

```python
# sample.py
"""Generate text from a trained KALIA checkpoint."""

import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig


def load_model(ckpt_path: str | Path, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"]["model"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def generate_text(
    ckpt_path: str | Path,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 200,
    encoder=None,
    device: torch.device | None = None,
) -> str:
    if encoder is None:
        import tiktoken

        encoder = tiktoken.get_encoding("gpt2")
    device = device or torch.device("cpu")
    model = load_model(ckpt_path, device)
    ids = encoder.encode_ordinary(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return encoder.decode(out[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    text = generate_text(
        args.ckpt,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=torch.device(args.device),
    )
    print(text)


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sample.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add sample.py tests/test_sample.py
git commit -m "feat(sample): checkpoint generation CLI"
```

---

### Task 11: Full local test suite + first green run

**Step 1: Run the entire suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass (~19 tests).

**Step 2: Fix anything red, rerun until green**

No commit needed unless fixes were made.

---

### Task 12: Kaggle notebooks

**Files:**
- Create: `notebooks/kalia-prep.ipynb`
- Create: `notebooks/kalia-train.ipynb`

**Step 1: Create `notebooks/kalia-prep.ipynb`**

A valid nbformat 4 JSON notebook with these cells (code cells unless noted):

1. Markdown: "# KALIA — data prep (CPU). Run with Save Version so outputs persist."
2. Code: `!pip install -q tiktoken datasets`
3. Code: `!git clone https://github.com/<YOUR_GH_USER>/kalia.git || (cd kalia && git pull)`
4. Code:
```python
%cd /kaggle/working/kalia
!python prepare.py --source tinystories --out /kaggle/working/data/train_stories.bin --max-tokens 500000000 --meta /kaggle/working/data/meta.json
!python prepare.py --source fineweb --out /kaggle/working/data/train_fineweb.bin --max-tokens 2000000000 --meta /kaggle/working/data/meta.json
!cat /kaggle/working/data/meta.json
```
5. Code (concatenate to train.bin, build val.bin):
```python
import numpy as np, json, shutil
from pathlib import Path
data = Path("/kaggle/working/data")
with open(data/"train_stories.bin","rb") as f: stories = f.read()
with open(data/"train_fineweb.bin","rb") as f: fineweb = f.read()
# val: 5M tokens from each source
val_stories, rest_stories = stories[:10_000_000], stories[10_000_000:]   # 5M uint16 = 10MB
val_fineweb, rest_fineweb = fineweb[:10_000_000], fineweb[10_000_000:]
with open(data/"val.bin","wb") as f: f.write(val_stories + val_fineweb)
with open(data/"train.bin","wb") as f: f.write(rest_stories + rest_fineweb)
(data/"train_stories.bin").unlink(); (data/"train_fineweb.bin").unlink()
print("train tokens:", (data/"train.bin").stat().st_size//2)
print("val tokens:", (data/"val.bin").stat().st_size//2)
```
6. Markdown: "Save Version → run in background. Then on the version page: Output → Create Dataset → name `kalia-tokens` (private)."

**Step 2: Create `notebooks/kalia-train.ipynb`**

Valid nbformat 4 JSON with these cells:

1. Markdown: "# KALIA — training (GPU T4×2). Set Accelerator: GPU T4 x2. Attach `kalia-tokens` dataset. Add HF_TOKEN to Kaggle Secrets."
2. Code: `!pip install -q tiktoken pyyaml huggingface_hub`
3. Code: `!git clone https://github.com/<YOUR_GH_USER>/kalia.git || (cd kalia && git pull)`
4. Code:
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["DATA_DIR"] = "/kaggle/input/kalia-tokens"
os.environ["OUT_DIR"] = "/kaggle/working/out"
```
5. Code (main training command — edit minutes if needed):
```python
%cd /kaggle/working/kalia
!torchrun --nproc_per_node=2 --standalone train.py \
  --config configs/kalia-m.yaml \
  --data-dir $DATA_DIR --out-dir $OUT_DIR \
  --resume --hub-repo <YOUR_HF_USER>/kalia-m --max-minutes 510
```
6. Markdown: "First run: --resume finds no checkpoint and starts from random init. Later runs continue. At session end, checkpoints are on HF Hub."

**Step 3: Validate notebook JSON**

Run:
```bash
python -c "import json; [json.load(open(f)) for f in ['notebooks/kalia-prep.ipynb','notebooks/kalia-train.ipynb']]; print('valid')"
```
Expected: `valid`

**Step 4: Commit**

```bash
git add notebooks/
git commit -m "feat(notebooks): Kaggle data-prep (CPU) and training (GPU T4x2)"
```

---

### Task 13: README with beginner glossary + runbook

**Files:**
- Modify: `README.md`

**Step 1: Write the README**

Sections (complete content):
- What KALIA is + the ownership statement (from-scratch, no fine-tuning).
- Beginner-friendly glossary table (weights, architecture, transformer, attention, embedding, tokenizer/BPE, context, RMSNorm/SwiGLU/RoPE, random init, training, loss, backprop, AdamW, LR/cosine/warmup, batch/step, fp16 AMP, DDP, checkpoint, overfit, fine-tune vs from-scratch, SFT, quantization/GGUF, Kaggle, HF Hub).
- Repo layout.
- Local quickstart: `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && python -m pytest tests/ -v`
- Kaggle runbook: (1) create private dataset `kalia-tokens` via kalia-prep, (2) create HF org `kalia` + private repo `kalia/kalia-m` + write token, (3) run kalia-train, (4) monitor via HF repo logs.
- Hardware reality: no local GPU — training on Kaggle only; local machine used for editing, tests, final inference.
- Roadmap v0–v3.
- Licenses: code MIT; weights to be released under the project's choice.

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with beginner glossary and Kaggle runbook"
```

---

## Manual account setup (not code tasks)

1. **HuggingFace**: register `kalia` org at https://huggingface.co/organizations/new (or any private repo), create a **write** token at https://huggingface.co/settings/tokens.
2. **Kaggle**: verify phone number (Settings → Phone Verification) to unlock GPU/TPU quotas.
3. **GitHub**: push this repo to a private GitHub repo (needed for cloning into Kaggle).
4. **Kaggle dataset**: after running `kalia-prep` with Save Version, create a private Dataset named `kalia-tokens` from the notebook output.
5. **Kaggle secret**: add `HF_TOKEN` in Kaggle → Add-ons → Secrets.
6. **First training run**: open `kalia-train`, accelerator GPU T4×2, run. Expected early log: loss starts ≈ ln(50257) ≈ 10.8 and falls toward ~4 within a few hundred steps (TinyStories dominates early).

## Success criteria

- `python -m pytest tests/ -v` fully green locally.
- Kaggle training resumes across sessions without losing progress (step count + tokens monotonically increase in `train_log.csv` on HF Hub).
- Final val loss ≈ 3.2–3.6; `python sample.py --ckpt ckpt.pt --prompt "Once upon a time"` produces coherent story-like English.
