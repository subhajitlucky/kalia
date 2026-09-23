from eval_reversibility import forward_and_reverse_nll, render_report
from model import GPT, GPTConfig


class FakeEncoder:
    eot_token = 255

    def encode_ordinary(self, text):
        return [ord(c) % 256 for c in text]


def _tiny_model(seed=0):
    import torch

    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, context_len=64))


def test_reversibility_metrics_finite():
    model = _tiny_model()
    result = forward_and_reverse_nll(
        model, FakeEncoder(), ["hello world", "the fox ran fast"]
    )
    assert result["forward_nll"] > 0
    assert result["reverse_nll"] > 0
    assert isinstance(result["abhimanyu_gap"], float)
    assert len(result["per_sentence"]) == 2
    for row in result["per_sentence"]:
        assert "quality_score" in row


def test_reversibility_deterministic():
    model = _tiny_model()
    first = forward_and_reverse_nll(model, FakeEncoder(), ["a short sentence"])
    second = forward_and_reverse_nll(model, FakeEncoder(), ["a short sentence"])
    assert first["forward_nll"] == second["forward_nll"]


def test_report_renders():
    model = _tiny_model()
    result = forward_and_reverse_nll(model, FakeEncoder(), ["hello world"])
    report = render_report(result)
    assert "Abhimanyu gap" in report
    assert "Forward NLL" in report
