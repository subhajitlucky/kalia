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
    assert set(SOURCES) == {
        "tinystories",
        "fineweb",
        "smollm_fineweb_edu",
        "cosmopedia",
        "stack_smol",
    }
