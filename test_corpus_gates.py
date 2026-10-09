"""Tests for the corpus contamination gate."""

import json

import numpy as np
import pytest

from corpus_gates import NGRAM, build_pattern_table, hash_window, scan_shard, window_hashes


class FakeEncoder:
    """Whitespace tokenizer with ids below 4096 so uint16 shards hold them."""

    @staticmethod
    def _tok(word: str) -> int:
        return sum(ord(c) for c in word) % 4096

    def encode_ordinary(self, text: str) -> list[int]:
        return [self._tok(word) for word in text.split()]


def write_shard(path, tokens):
    np.array(tokens, dtype=np.uint16).tofile(path)
    return path


def test_hash_window_matches_the_vectorized_hashes():
    window = tuple(range(100, 100 + NGRAM))
    vector = window_hashes(np.array(window, dtype=np.int64))
    assert int(vector[0]) == hash_window(window)


def test_injected_pattern_is_found_at_the_right_position(tmp_path):
    encoder = FakeEncoder()
    doc = {"task": "unit", "index": 0, "text": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"}
    pattern_ids = encoder.encode_ordinary(doc["text"])
    filler = [7] * 40
    tokens = filler + pattern_ids + filler
    shard = write_shard(tmp_path / "train.bin", tokens)

    hashes, patterns = build_pattern_table(encoder, [doc])
    result = scan_shard(shard, hashes, patterns, chunk=16)

    assert result["per_task"] == {"unit": 1}
    assert result["hits"][0]["token_index"] == 40
    assert result["hits"][0]["tokens"] == pattern_ids


def test_pattern_across_a_chunk_boundary_is_not_missed(tmp_path):
    encoder = FakeEncoder()
    doc = {"task": "unit", "index": 0, "text": "one two three four five six seven eight nine ten eleven twelve thirteen"}
    pattern_ids = encoder.encode_ordinary(doc["text"])
    tokens = [7] * 9 + pattern_ids + [7] * 9
    shard = write_shard(tmp_path / "train.bin", tokens)

    hashes, patterns = build_pattern_table(encoder, [doc])
    result = scan_shard(shard, hashes, patterns, chunk=10)

    assert result["per_task"] == {"unit": 1}
    assert result["hits"][0]["token_index"] == 9


def test_absent_patterns_produce_no_hits(tmp_path):
    encoder = FakeEncoder()
    doc = {"task": "unit", "index": 0,
           "text": "never appears in the shard at all anywhere no really it does not appear here"}
    shard = write_shard(tmp_path / "train.bin", [3] * 50)

    hashes, patterns = build_pattern_table(encoder, [doc])
    result = scan_shard(shard, hashes, patterns, chunk=16)

    assert result["per_task"] == {}
    assert result["hits"] == []


def test_short_shards_are_handled(tmp_path):
    encoder = FakeEncoder()
    doc = {"task": "unit", "index": 0, "text": "a b c d e f g h i j k l m n o p"}
    shard = write_shard(tmp_path / "tiny.bin", [1, 2, 3])

    hashes, patterns = build_pattern_table(encoder, [doc])
    result = scan_shard(shard, hashes, patterns)
    assert result["scanned_tokens"] == 0 or result["per_task"] == {}


def test_degenerate_windows_are_skipped_not_scanned_for(tmp_path):
    encoder = FakeEncoder()
    degenerate = {"task": "unit", "index": 0, "text": " ".join(["spam"] * 14)}
    with pytest.raises(ValueError):
        build_pattern_table(encoder, [degenerate])

    real = {"task": "unit", "index": 1,
            "text": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"}
    pattern_ids = encoder.encode_ordinary(real["text"])
    shard = write_shard(tmp_path / "train.bin", [7] * 20 + pattern_ids + [7] * 20)
    hashes, patterns = build_pattern_table(encoder, [real])
    result = scan_shard(shard, hashes, patterns, chunk=16)
    assert result["per_task"] == {"unit": 1}


def test_hashes_are_stable_for_the_same_window():
    window = tuple(int(t) % 50257 for t in range(1000, 1000 + NGRAM))
    assert hash_window(window) == hash_window(window)
    ids = np.array(window, dtype=np.int64)
    assert int(window_hashes(ids)[0]) == hash_window(window)


def test_build_pattern_table_requires_content():
    encoder = FakeEncoder()
    with pytest.raises(ValueError):
        build_pattern_table(encoder, [])
    with pytest.raises(ValueError):
        build_pattern_table(encoder, [{"task": "tiny", "text": "too short"}])
