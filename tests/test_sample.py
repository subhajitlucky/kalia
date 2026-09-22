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
