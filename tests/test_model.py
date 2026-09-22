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
