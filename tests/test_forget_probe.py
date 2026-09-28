"""Tests for the forgetting-probe instrumentation (CL-0)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

SEP = 50256


def _load(name: str):
    path = Path(__file__).resolve().parent.parent / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


builder = _load("build_forget_probe")


def _corpus(n_docs: int = 60, doc_len: int = 37) -> np.ndarray:
    rng = np.random.default_rng(0)
    docs = []
    for _ in range(n_docs):
        toks = rng.integers(0, 50000, size=doc_len).astype(np.uint16)
        docs.append(np.concatenate([toks, np.array([SEP], dtype=np.uint16)]))
    return np.concatenate(docs)


def test_split_lands_on_a_document_boundary():
    tokens = _corpus()
    cut = builder.split_at_document(tokens, 0.5)
    assert tokens[cut - 1] == SEP, "cut must fall immediately after a separator"
    assert 0 < cut < len(tokens)


def test_split_halves_are_both_valid_document_streams():
    tokens = _corpus()
    cut = builder.split_at_document(tokens, 0.5)
    head, tail = tokens[:cut], tokens[cut:]
    for name, part in (("head", head), ("tail", tail)):
        assert part.size, f"{name} is empty"
        assert part[-1] == SEP, f"{name} must end on a document boundary"
        assert part[0] != SEP, f"{name} must not start mid-document"
        assert (part == SEP).sum() > 0, f"{name} has no documents"


def test_split_preserves_all_documents_exactly_once():
    """A dropped or duplicated document would silently change what the probe measures."""
    tokens = _corpus()
    cut = builder.split_at_document(tokens, 0.5)
    head, tail = tokens[:cut], tokens[cut:]
    assert len(head) + len(tail) == len(tokens)
    assert (head == SEP).sum() + (tail == SEP).sum() == (tokens == SEP).sum()


def test_describe_reports_document_statistics():
    tokens = _corpus(n_docs=10, doc_len=20)
    d = builder.describe(tokens)
    assert d["tokens"] == len(tokens)
    assert d["documents"] == 10
    assert d["median_doc_len"] > 0
    assert d["ends_with_sep"] is True


def test_probe_refuses_to_split_an_unmarked_file():
    """Without separators there is no safe cut, and guessing would corrupt both halves."""
    tokens = np.arange(1000, dtype=np.uint16)  # no separators at all
    try:
        builder.split_at_document(tokens, 0.5)
    except ValueError as exc:
        assert "separator" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a file with no separators")


def test_ledger_roundtrip(tmp_path):
    """The ledger is append-only JSONL; a malformed line would break every later read."""
    probe = _load("forgetting_probe")
    path = tmp_path / "forgetting.jsonl"
    assert probe.read_ledger(path) == []
    rows = [{"step": 100, "probe_loss": 2.5}, {"step": 200, "probe_loss": 2.4}]
    with path.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    assert probe.read_ledger(path) == rows
    # A trailing blank line must not produce a phantom record.
    with path.open("a") as fh:
        fh.write("\n")
    assert len(probe.read_ledger(path)) == 2


# ------------------------------------------------- replay shard construction

# make_replay_shard.py lives at the repo root, next to mix_bins.py and prepare.py,
# because it is part of the kernel-time data pipeline rather than a dev tool.
_root = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("make_replay_shard", _root / "make_replay_shard.py")
sharder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sharder)


def test_blocks_are_contiguous_non_overlapping_and_spread():
    total = 10_000_000
    ranges = sharder.choose_blocks(total, target_tokens=1_000_000, n_blocks=32, seed=7)
    assert len(ranges) == 32
    seen = []
    for start, length in ranges:
        assert start >= 0 and start + length <= total, "block falls outside the file"
        seen.append((start, start + length))
    seen.sort()
    for (_, a_end), (b_start, _) in zip(seen, seen[1:]):
        assert a_end <= b_start, "blocks overlap"
    # Spread: the first and last blocks should be far apart.
    assert max(s for s, _ in ranges) - min(s for s, _ in ranges) > total // 2


def test_blocks_are_deterministic_for_a_seed():
    a = sharder.choose_blocks(1_000_000, 100_000, 8, seed=3)
    b = sharder.choose_blocks(1_000_000, 100_000, 8, seed=3)
    c = sharder.choose_blocks(1_000_000, 100_000, 8, seed=4)
    assert a == b, "same seed must give the same shard"
    assert a != c, "different seeds must give different shards"


def test_target_larger_than_source_is_rejected():
    try:
        sharder.choose_blocks(1_000, target_tokens=5_000, n_blocks=4, seed=1)
    except SystemExit:
        pass
    else:
        raise AssertionError("expected a SystemExit when the target exceeds the file")


def test_align_to_documents_yields_whole_documents():
    toks = np.array([5, 6, SEP, 7, 8, 9, SEP, 4, 5], dtype=np.uint16)
    out = sharder.align_to_documents(toks)
    assert out[0] == SEP, "must start on a document boundary"
    assert out[-1] == SEP, "must end on a document boundary"


def test_carved_shard_is_valid_and_recorded(tmp_path):
    """End-to-end on a small synthetic corpus: the shard must be a real token stream
    and the manifest must record the ranges the daily pipeline has to avoid."""
    import json
    import subprocess
    import sys

    rng = np.random.default_rng(0)
    docs = []
    for _ in range(4000):
        body = rng.integers(0, 50000, size=60).astype(np.uint16)
        docs.append(np.concatenate([body, np.array([SEP], dtype=np.uint16)]))
    corpus = np.concatenate(docs)
    src = tmp_path / "train.bin"
    corpus.tofile(src)

    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_root / "make_replay_shard.py"),
         "--train-bin", str(src), "--out-dir", str(out), "--fraction", "0.2", "--blocks", "4",
         "--seed", "1337"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-500:]
    shard = np.memmap(out / "replay.bin", dtype=np.uint16, mode="r")
    assert len(shard) > 0
    assert shard[0] == SEP and shard[-1] == SEP, "shard must be whole documents"
    manifest = json.loads((out / "replay_manifest.json").read_text())
    assert manifest["replay_tokens"] == len(shard)
    assert len(manifest["blocks"]) == 4
    assert manifest["source_tokens"] == len(corpus)
    assert "EXCLUDE" in manifest["purpose"]
