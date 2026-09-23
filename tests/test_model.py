import torch
from model import (
    RMSNorm,
    rope_tables,
    apply_rope,
    SwiGLU,
    GPTConfig,
    CausalSelfAttention,
    GPT,
)


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


def test_qk_norm_adds_only_norm_params():
    base = GPT(GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, context_len=16))
    normed = GPT(
        GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, context_len=16, qk_norm=True)
    )
    head_dim = 64 // 2
    assert normed.num_params() - base.num_params() == 2 * head_dim * 2


def test_qk_norm_forward_backward():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, context_len=16, qk_norm=True)
    model = GPT(cfg)
    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    _, loss = model(x, y)
    loss.backward()
    assert torch.isfinite(loss)


def test_logit_softcap_bounds_logits():
    torch.manual_seed(0)
    cfg = GPTConfig(
        vocab_size=256, n_layer=2, n_head=2, n_embd=64, context_len=16, logit_softcap=5.0
    )
    model = GPT(cfg)
    x = torch.randint(0, 256, (2, 8))
    logits, _ = model(x)
    assert logits.abs().max().item() <= 5.0 + 1e-4


def test_gqa_reduces_params_and_runs():
    base = GPT(GPTConfig(vocab_size=256, n_layer=2, n_head=4, n_embd=64, context_len=16))
    gqa = GPT(
        GPTConfig(vocab_size=256, n_layer=2, n_head=4, n_embd=64, context_len=16, n_kv_head=2)
    )
    assert gqa.num_params() < base.num_params()
    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    _, loss = gqa(x, y)
    loss.backward()
    assert torch.isfinite(loss)


def test_gqa_equal_heads_matches_mha_params():
    base = GPT(GPTConfig(vocab_size=256, n_layer=2, n_head=4, n_embd=64, context_len=16))
    same = GPT(
        GPTConfig(vocab_size=256, n_layer=2, n_head=4, n_embd=64, context_len=16, n_kv_head=4)
    )
    assert same.num_params() == base.num_params()


def test_looping_shares_params_and_runs():
    once = GPT(GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, context_len=16))
    twice = GPT(
        GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, context_len=16, n_loops=2)
    )
    assert once.num_params() == twice.num_params()
    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    _, loss = twice(x, y)
    loss.backward()
    assert torch.isfinite(loss)


def test_looping_changes_output_not_params():
    import copy

    torch.manual_seed(0)
    base_cfg = GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, context_len=16)
    model = GPT(base_cfg)
    looped = copy.deepcopy(model)
    looped.cfg = GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, context_len=16, n_loops=2)
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        logits_base, _ = model(x)
        logits_looped, _ = looped(x)
    assert not torch.allclose(logits_base, logits_looped)
