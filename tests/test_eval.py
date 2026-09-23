import torch
from compare_models import compare
from eval_probes import build_report, generate, load_prompts, load_sentences, loss_and_bpb
from model import GPT, GPTConfig


class FakeEncoder:
    eot_token = 255

    def encode_ordinary(self, text):
        return [ord(c) % 256 for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _tiny_ckpt(tmp_path, name="ckpt.pt", seed=0):
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, context_len=64)
    model = GPT(cfg)
    path = tmp_path / name
    torch.save(
        {
            "model": model.state_dict(),
            "step": 10,
            "tokens": 5120,
            "config": {"model": cfg.__dict__, "train": {}},
        },
        path,
    )
    return path, model


def test_probe_files_load():
    prompts = load_prompts("eval/probe_prompts.json")
    sentences = load_sentences("eval/probe_sentences.json")
    assert len(prompts) == 24
    assert len(sentences) == 20
    assert all(isinstance(p, str) and p for p in prompts)


def test_generate_is_deterministic(tmp_path):
    _, model = _tiny_ckpt(tmp_path)
    encoder = FakeEncoder()
    first = generate(model, encoder, "Once", max_new_tokens=10, seed=7)
    second = generate(model, encoder, "Once", max_new_tokens=10, seed=7)
    assert first == second
    assert len(first) == 4 + 10


def test_loss_and_bpb_are_finite(tmp_path):
    _, model = _tiny_ckpt(tmp_path)
    loss, bpb = loss_and_bpb(model, FakeEncoder(), ["hello world", "the fox ran"])
    assert 0 < loss < 20
    assert bpb > 0


def test_build_report_contains_fields():
    report = build_report(
        {"step": 10, "tokens": 5120},
        {"Once": "Once upon"},
        3.5,
        1.25,
        {"seed": 1, "temperature": 0.8, "top_k": 200},
    )
    assert "KALIA Evaluation Report" in report
    assert "3.5000" in report
    assert "1.2500" in report
    assert "Once upon" in report


def test_compare_writes_markdown(tmp_path):
    path_a, _ = _tiny_ckpt(tmp_path, name="a.pt", seed=1)
    path_b, _ = _tiny_ckpt(tmp_path, name="b.pt", seed=2)
    out = tmp_path / "comparison.md"
    compare(
        path_a,
        path_b,
        ["Once"],
        FakeEncoder(),
        out,
        label_a="baseline",
        label_b="upgraded",
        max_new_tokens=5,
    )
    text = out.read_text()
    assert "baseline" in text and "upgraded" in text
