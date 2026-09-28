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
