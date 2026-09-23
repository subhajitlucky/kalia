import numpy as np
from mix_bins import mix_shards


def _write(path, values):
    np.array(values, dtype=np.uint16).tofile(path)


def test_mix_shards_ratio_and_val_split(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    _write(a, list(range(0, 1000)))  # source A: values 0..999
    _write(b, list(range(1000, 2000)))  # source B: values 1000..1999

    val_out = tmp_path / "val.bin"
    train_out = tmp_path / "train.bin"
    stats = mix_shards(
        [(a, 3.0), (b, 1.0)],
        val_out,
        train_out,
        val_tokens=200,
        train_tokens=800,
        block_tokens=100,
    )

    assert stats["val_tokens"] == 200
    assert stats["train_tokens"] == 800

    val = np.fromfile(val_out, dtype=np.uint16)
    assert len(val) == 200
    from_b_val = (val >= 1000).sum()
    assert 40 <= from_b_val <= 60  # ~25% of val from source B

    train = np.fromfile(train_out, dtype=np.uint16)
    assert len(train) == 800
    from_b_train = (train >= 1000).sum()
    assert 180 <= from_b_train <= 220  # ~25% of train from source B
    # train must not overlap the tokens consumed by val
    assert train.min() >= 0


def test_mix_shards_stops_when_sources_exhausted(tmp_path):
    a = tmp_path / "a.bin"
    _write(a, list(range(0, 50)))
    stats = mix_shards([(a, 1.0)], tmp_path / "val.bin", tmp_path / "train.bin", 40, 100, 10)
    assert stats["val_tokens"] == 40
    assert stats["train_tokens"] == 10  # only 10 tokens remained
