"""Lexical lookup over a local corpus, so the model can read what it cannot store.

A 58M-parameter model has no room for facts; with 44.5% of its weights inside
the embedding it has even less than the parameter count suggests. The
deployment-side answer the project's research settled on is retrieval: keep
knowledge in an index, look up the relevant passages when a question arrives,
and let the model read them. This module is the lookup half — no external
weights, no training, pure Python plus the standard library.

BM25 is the ranking function, for two reasons that matter to this project:

- it needs no embeddings, so D2 (no borrowed weights, ever) is untouched;
- its failure mode is honest and visible — it misses paraphrases — unlike a
  borrowed encoder's, which would look fluent while quietly importing someone
  else's training distribution.

A dense retriever is a later, separately registered question. This one is the
floor: lexical, deterministic, and cheap enough to run anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Punctuation never carries a match."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 over a fixed list of documents.

    Parameters are the standard defaults (k1 = 1.5, b = 0.75). They are exposed
    rather than buried because the right setting is corpus-dependent, and a
    tuning claim would need its own measurement.
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        if not docs:
            raise ValueError("cannot build an index over an empty corpus")
        if any(not doc.strip() for doc in docs):
            raise ValueError("empty documents are not indexable; filter them first")
        self.docs = list(docs)
        self.k1 = float(k1)
        self.b = float(b)
        self._tfs = [Counter(tokenize(doc)) for doc in self.docs]
        self._lengths = [sum(tf.values()) for tf in self._tfs]
        self._avgdl = sum(self._lengths) / len(self._lengths)
        df: Counter = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(self.docs)
        self._idf = {
            term: math.log(1.0 + (n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        tf = self._tfs[doc_index]
        dl = self._lengths[doc_index]
        score = 0.0
        for term in set(query_terms):
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = self._idf.get(term, 0.0)
            denom = freq + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
            score += idf * freq * (self.k1 + 1.0) / denom
        return score

    def retrieve(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        """Top ``k`` (document index, score) pairs, best first.

        Zero-score documents are omitted: a result that matched no term is not
        a weak match, and returning it anyway would let a caller quote a
        retrieval where none happened. Ties break to the lower index, so the
        same query always returns the same order.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        terms = tokenize(query)
        if not terms:
            return []
        scored = [
            (i, self._score(terms, i)) for i in range(len(self.docs))
        ]
        hits = [(i, s) for i, s in scored if s > 0.0]
        hits.sort(key=lambda item: (-item[1], item[0]))
        return hits[:k]

    def retrieve_docs(self, query: str, k: int = 3) -> list[str]:
        return [self.docs[i] for i, _ in self.retrieve(query, k)]


def build_context(query: str, index: BM25Index, k: int = 3, max_chars: int = 2000) -> str:
    """Join the top documents, newest-ranked first, within a character budget.

    Documents are taken whole while they fit; if the top hit alone exceeds the
    budget it is truncated rather than dropped, because the caller asked for a
    lookup and an empty string would look like the index found nothing.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars}")
    hits = index.retrieve(query, k)
    if not hits:
        return ""
    parts: list[str] = []
    used = 0
    for i, _ in hits:
        doc = index.docs[i]
        if not parts:
            parts.append(doc[:max_chars])
            used = len(parts[0])
            continue
        if used + 1 + len(doc) > max_chars:
            break
        parts.append(doc)
        used += 1 + len(doc)
    return "\n".join(parts)


def augment_prompt(query: str, index: BM25Index, k: int = 3, max_chars: int = 2000) -> str:
    """The prompt a caller would send: retrieved context, blank line, question.

    Plain concatenation, deliberately. KALIA was trained on raw text, not on
    ``[context]`` markers, so inventing a marker vocabulary the model has never
    seen would be one more untested variable on top of an untested mechanism.
    If a separator format is to earn its place, it earns it by measurement.
    """
    context = build_context(query, index, k=k, max_chars=max_chars)
    if not context:
        return query
    return f"{context}\n\n{query}"


def load_corpus(path: Path) -> list[str]:
    """One document per line for .txt, or one ``{"text": ...}`` object per line for .jsonl."""
    if path.suffix == ".jsonl":
        docs = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                docs.append(str(json.loads(line)["text"]))
        return docs
    return [line for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=2000)
    args = parser.parse_args()

    index = BM25Index(load_corpus(args.corpus))
    for i, score in index.retrieve(args.query, args.k):
        print(f"[{score:6.3f}] {index.docs[i][:100]}")
    print()
    print(augment_prompt(args.query, index, k=args.k, max_chars=args.max_chars))


if __name__ == "__main__":
    main()
