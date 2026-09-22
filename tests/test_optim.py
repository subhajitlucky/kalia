import torch
from model import GPT, GPTConfig
from optim import (
    Muon,
    MuonWithAuxAdam,
    split_muon_params,
    zeroth_power_via_newtonschulz5,
)
from train import lr_scale


def test_ns5_produces_near_orthogonal_matrix():
    torch.manual_seed(0)
    g = torch.randn(32, 32)
    out = zeroth_power_via_newtonschulz5(g, steps=5)
    assert out.shape == g.shape
    singular_values = torch.linalg.svdvals(out.float())
    # The quintic iteration converges to singular values in ~[0.68, 1.13]
    assert singular_values.min() > 0.5
    assert singular_values.max() < 1.4


def test_ns5_handles_tall_and_wide_matrices():
    for shape in [(64, 16), (16, 64)]:
        torch.manual_seed(0)
        g = torch.randn(*shape)
        out = zeroth_power_via_newtonschulz5(g, steps=5)
        assert out.shape == g.shape
        sv = torch.linalg.svdvals(out.float())
        assert sv.min() > 0.5 and sv.max() < 1.4


def test_muon_optimizes_quadratic_problem():
    torch.manual_seed(0)
    target = torch.randn(8, 8)
    weight = torch.nn.Parameter(torch.zeros(8, 8))
    optimizer = Muon([weight], lr=0.05)
    first_loss = None
    last_loss = None
    for _ in range(100):
        optimizer.zero_grad()
        loss = ((weight - target) ** 2).sum()
        loss.backward()
        optimizer.step()
        first_loss = first_loss if first_loss is not None else loss.item()
        last_loss = loss.item()
    assert last_loss < first_loss * 0.1, (first_loss, last_loss)


def _tiny_model() -> GPT:
    return GPT(GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, context_len=16))


def test_split_muon_params_partitions_everything_once():
    model = _tiny_model()
    hidden, other = split_muon_params(model)
    assert all(p.ndim >= 2 for p in hidden)
    hidden_names = {
        name for name, p in model.named_parameters() if any(p is q for q in hidden)
    }
    assert not any(name.startswith("tok_emb") or name.startswith("lm_head") for name in hidden_names)
    assert any("qkv.weight" in name for name in hidden_names)
    assert any("down.weight" in name for name in hidden_names)
    total_params = sum(p.numel() for p in hidden) + sum(p.numel() for p in other)
    assert total_params == model.num_params()


def test_muon_with_aux_adam_steps_and_checkpoints():
    torch.manual_seed(0)
    model = _tiny_model()
    hidden, other = split_muon_params(model)
    optimizer = MuonWithAuxAdam(hidden, other, muon_lr=0.02, adam_lr=6e-4)
    assert len(optimizer.param_groups) == 2
    base_lrs = {group["base_lr"] for group in optimizer.param_groups}
    assert base_lrs == {0.02, 6e-4}

    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)

    state = optimizer.state_dict()
    optimizer.load_state_dict(state)


def test_lr_scale_schedule_shape():
    cfg = {"learning_rate": 1e-3, "warmup_steps": 10, "max_steps": 110, "min_lr_ratio": 0.1}
    assert 0 < lr_scale(0, cfg) < lr_scale(9, cfg)
    assert lr_scale(9, cfg) == 1.0
    assert 0.1 < lr_scale(50, cfg) < 1.0
    assert lr_scale(110, cfg) == 0.1
