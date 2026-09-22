import torch
from model import RMSNorm, rope_tables, apply_rope, SwiGLU, GPTConfig, CausalSelfAttention


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


def test_swiglu_shape():
    cfg = GPTConfig(n_embd=64)
    mlp = SwiGLU(cfg)
    x = torch.randn(2, 5, 64)
    assert mlp(x).shape == x.shape
    hidden = mlp.gate.out_features
    assert hidden % 64 == 0  # hidden dim rounded to multiple of 64


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
