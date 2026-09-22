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
