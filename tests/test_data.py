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


def test_segment_permutation_fixed_chunks(tmp_path):
    path = tmp_path / "train.bin"
    _write_bin(path, list(range(1000)))
    ds = TokenDataset(
        path, context_len=8, reversal_prob=1.0, reversal_min_chunk=4, reversal_max_chunk=4
    )
    perm = ds._segment_permutation(8, torch.Generator().manual_seed(0))
    assert perm.tolist() == [4, 5, 6, 7, 0, 1, 2, 3]


def test_reversal_chunk_structure(tmp_path):
    path = tmp_path / "train.bin"
    _write_bin(path, list(range(1000)))
    ds = TokenDataset(
        path, context_len=16, reversal_prob=1.0, reversal_min_chunk=4, reversal_max_chunk=4
    )
    x, _ = ds.get_batch(2, torch.device("cpu"), torch.Generator().manual_seed(1))
    row = x[0]
    breaks = [i for i in range(15) if row[i + 1] != row[i] + 1]
    assert breaks == [3, 7, 11]  # four chunks of 4, order reversed
    assert row[0] > row[15]  # the original last chunk now comes first


def test_reversal_keeps_targets_aligned(tmp_path):
    path = tmp_path / "train.bin"
    _write_bin(path, list(range(1000)))
    ds = TokenDataset(
        path, context_len=16, reversal_prob=1.0, reversal_min_chunk=4, reversal_max_chunk=8
    )
    x, y = ds.get_batch(4, torch.device("cpu"), torch.Generator().manual_seed(2))
    assert torch.equal(y, x + 1)  # consecutive-id file: successor invariant holds


def test_reversal_off_is_identity(tmp_path):
    path = tmp_path / "train.bin"
    _write_bin(path, list(range(1000)))
    ds = TokenDataset(path, context_len=16)
    x, y = ds.get_batch(4, torch.device("cpu"), torch.Generator().manual_seed(3))
    assert torch.equal(y, x + 1)
    assert torch.equal(y[:, :-1], x[:, 1:])
