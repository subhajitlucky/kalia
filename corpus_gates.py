"""Corpus data gates: a 13-gram benchmark-contamination scan at token level.

L20-Edu-135M (arXiv 2606.22189, title-verified) treats contamination checks as a
release gate rather than as analysis: candidate training data is rejected when a
13-gram matches the evaluation sets. KALIA's corpus predates any such check, and
its public card should not rest on the assumption that FineWeb-Edu simply never
contains benchmark passages.

This module scans tokenized shards for **exact** 13-gram matches against
benchmark texts. Exact means exact: every hash hit is verified by comparing the
tokens themselves, so a hash collision can never appear as a finding. The scan
is deterministic and shard positions are recorded, so any hit can be re-read
from the raw file.

Hashing is polynomial with uint64 wraparound (base 1,000,003). Wraparound is
fine for a hash; verification is what makes the result a fact, not the hash
function. Windowed hashing is vectorized over 1M-token chunks so a 2.4B-token
shard is minutes, not days, on one CPU.

Usage:
    python corpus_gates.py --benchmark-jsonl bench.jsonl \\
        --shard data/train.bin --shard data/val.bin --out gates.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

NGRAM = 13
BASE = 1_000_003
MASK64 = (1 << 64) - 1
POWERS = np.array([pow(BASE, k, 1 << 64) for k in range(NGRAM)], dtype=np.uint64)

# A pattern window must carry at least this many distinct token ids to be worth
# scanning for. The first full scan reported 63,491,977 "verified" PIQA hits in
# 2.4B tokens -- all of them whitespace: 120 PIQA rows carry copied web text
# with long space runs, and a 13-space window matches Python indentation
# millions of times. A count that large was the bug report. Six distinct tokens
# in thirteen is trivially satisfied by any real sentence and rules out runs,
# padding, and separator art. Corpus windows are not filtered -- only patterns,
# so a degenerate corpus window can never match a filtered-out pattern.
MIN_DISTINCT = 6


def hash_window(window: tuple[int, ...]) -> int:
    """Hash one window exactly as :func:`window_hashes` does, for verification."""
    if len(window) != NGRAM:
        raise ValueError(f"window must have exactly {NGRAM} tokens, got {len(window)}")
    total = 0
    for k, token in enumerate(window):
        total = (total + int(token) * pow(BASE, k, 1 << 64)) & MASK64
    return total


def window_hashes(tokens: np.ndarray) -> np.ndarray:
    """Vectorized hashes for every 13-gram of ``tokens``."""
    if tokens.size < NGRAM:
        return np.empty(0, dtype=np.uint64)
    windows = np.lib.stride_tricks.sliding_window_view(tokens, NGRAM)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (windows.astype(np.uint64) * POWERS).sum(axis=1)


def build_pattern_table(encoder, docs: list[dict]) -> tuple[np.ndarray, dict[int, list[tuple]]]:
    """Sorted pattern hashes plus hash -> [(task, doc index, token window)].

    The table carries the token windows themselves so a corpus hit can be
    verified by equality rather than trusted on the hash. Windows with fewer
    than ``MIN_DISTINCT`` distinct tokens are skipped (see the constant).
    """
    table: dict[int, list[tuple]] = {}
    skipped = 0
    for doc in docs:
        ids = encoder.encode_ordinary(doc["text"])
        for start in range(max(0, len(ids) - NGRAM + 1)):
            window = tuple(int(t) for t in ids[start : start + NGRAM])
            if len(set(window)) < MIN_DISTINCT:
                skipped += 1
                continue
            table.setdefault(hash_window(window), []).append(
                (doc["task"], int(doc.get("index", 0)), window)
            )
    if skipped:
        print(f"degenerate windows skipped: {skipped}")
    if not table:
        raise ValueError("no 13-grams in any benchmark document; nothing to scan for")
    return np.array(sorted(table), dtype=np.uint64), table


def scan_shard(
    path: Path,
    pattern_hashes: np.ndarray,
    patterns: dict[int, list[tuple]],
    chunk: int = 1_000_000,
) -> dict:
    """Scan one shard; return verified hits and per-task counts.

    A hit is only recorded after the full 13 tokens compare equal, so the
    counts are lower bounds on real contamination, never hash artifacts.
    """
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    if len(tokens) < NGRAM:
        return {"shard": str(path), "hits": [], "per_task": {}, "scanned_tokens": 0}
    hits: list[dict] = []
    per_task: dict[str, int] = {}
    for start in range(0, len(tokens), chunk):
        piece = np.asarray(tokens[start : start + chunk + NGRAM - 1], dtype=np.int64)
        if piece.size < NGRAM:
            break
        hashes = window_hashes(piece)
        idx = np.searchsorted(pattern_hashes, hashes)
        clipped = np.clip(idx, 0, len(pattern_hashes) - 1)
        candidates = (idx < len(pattern_hashes)) & (pattern_hashes[clipped] == hashes)
        for pos in np.nonzero(candidates)[0]:
            window = tuple(int(t) for t in piece[pos : pos + NGRAM])
            for task, doc_index, pattern_window in patterns.get(int(hashes[pos]), []):
                if window != pattern_window:
                    continue
                per_task[task] = per_task.get(task, 0) + 1
                if len(hits) < 5000:
                    hits.append(
                        {
                            "token_index": int(start + pos),
                            "task": task,
                            "doc_index": doc_index,
                            "tokens": list(window),
                        }
                    )
    return {
        "shard": str(path),
        "scanned_tokens": int(len(tokens)),
        "hits": hits,
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-jsonl", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=1_000_000)
    args = parser.parse_args()

    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    docs = [
        json.loads(line)
        for line in args.benchmark_jsonl.read_text().splitlines()
        if line.strip()
    ]
    print(f"benchmark documents: {len(docs)}")
    pattern_hashes, patterns = build_pattern_table(encoder, docs)
    print(f"distinct 13-gram hashes: {len(pattern_hashes):,}")

    report: dict = {"ngram": NGRAM, "shards": []}
    for shard in args.shard:
        result = scan_shard(shard, pattern_hashes, patterns, chunk=args.chunk)
        report["shards"].append(result)
        total = sum(result["per_task"].values())
        print(f"{shard}: {result['scanned_tokens']:,} tokens | "
              f"{total} verified 13-gram occurrences {result['per_task']}")
    report["total_hits"] = sum(
        sum(shard["per_task"].values()) for shard in report["shards"]
    )
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\ntotal verified occurrences: {report['total_hits']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
