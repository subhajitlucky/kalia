"""Long-form source for v0.2.1 — the LAMBADA fix, with the hypothesis verified.

The problem, measured on our own v2b val set:

    p50 document length      239 tokens
    p90                     1066
    documents >= 1024          10.8%

With `context_len = 1024` and a median document of 239 tokens, almost no
training window is a complete document. The model is trained mostly on fragments
stitched across boundaries -- and I15 already showed it attends across those
boundaries. LAMBADA asks for the opposite skill: hold a broad discourse in
context and recall the final word. We are grading a long-range ability on a
corpus that almost never contains long-range structure. That is why LAMBADA sits
at 18.4%, barely above the ~10% a model scores picking a random capitalised
word.

The fix is data, not architecture: put real long documents in the mixture. PG-19
(DeepMind) is books published before 1919, chosen specifically for being out of
copyright, so it is licence-clean under the same standard we applied to the code
shard. Books run to tens of thousands of tokens, so a 1024-token window lands
*inside* a single narrative rather than straddling four of them.

What this script does
---------------------
1. Measures document-length distribution per source, so the claim above is
   re-verified on the actual training mixture rather than assumed.
2. Streams a long-form source and reports what it would add.

Run as a Kaggle **CPU** kernel: no GPU quota.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator

DOC_SEPARATOR = 50256
# (label, repo_id, config, field). Repo ids and configs are distinct: the two
# smollm-corpus sources share one repo and differ only by config name.
#
# The long-form source is `sedthh/gutenberg_english`, not `deepmind/pg19`. Two
# reasons, both checked rather than assumed:
#
#   1. pg19 is script-based (a `pg19.py` loader plus Gutenberg file-id lists, no
#      parquet at all) and modern `datasets` refuses to run dataset scripts, so
#      pg19 cannot be loaded at all -- streaming or not. The first run died on
#      exactly this.
#   2. The obvious pg19 parquet mirrors declare **no licence** (`emozilla/pg19`
#      and `manu/project_gutenberg` both return license=None). Under D44 a
#      source with no declared licence fails our own permissive filter, so
#      neither is admissible no matter how well it matches the original corpus.
#
# `sedthh/gutenberg_english` is 37 parquet files and declares MIT. Caveat worth
# carrying into any decision to actually train on it: the MIT tag covers the
# dataset packaging, while the underlying Gutenberg texts are public domain in
# the US but not uniformly so in every jurisdiction. That is a judgement call
# for the adoption decision, not something a probe should decide silently.
SOURCES = [
    ("cosmopedia-v2", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text"),
    ("fineweb-edu-dedup", "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", "text"),
    ("TinyStories", "roneneldan/TinyStories", None, "text"),
    ("gutenberg-english (long-form)", "sedthh/gutenberg_english", None, "TEXT"),
]


def _pick_field(row: dict, field: str) -> str:
    """Resolve the text column, tolerating case differences.

    `sedthh/gutenberg_english` names its column `TEXT` while every other source
    here uses `text`. Guessing wrong is not a crash -- `row.get("text")` returns
    None, the `if text` guard drops every row, and the probe reports "0
    documents" as if that were a measurement. So a missing field is raised
    loudly instead of silently emptying the sample.
    """
    if field in row:
        return field
    lowered = {k.lower(): k for k in row}
    if field.lower() in lowered:
        return lowered[field.lower()]
    for alias in ("text", "content", "TEXT", "Content"):
        if alias in row:
            return alias
    raise KeyError(
        f"text column not found: wanted {field!r}, dataset has {sorted(row)[:8]}"
    )


def iter_texts(repo: str, config: str | None, field: str, limit: int | None = None) -> Iterator[str]:
    from datasets import load_dataset

    if config:
        ds = load_dataset(repo, name=config, split="train", streaming=True)
    else:
        # Long-form sources are large; stream them and stop at `limit` rather
        # than materialising the whole split. The first run SIGABRT'd on a
        # non-streaming read of a multi-GB script dataset.
        ds = load_dataset(repo, split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        text = row.get(_pick_field(row, field)) or ""
        if text:
            yield text


def length_stats(texts: Iterable[str], limit: int = 3000) -> dict:
    import numpy as np
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    lens: list[int] = []
    for text in texts:
        n = len(enc.encode(text, disallowed_special=()))
        if n < 64:  # ignore stubs; they cannot be a 1024-token window
            continue
        lens.append(n)
        if len(lens) >= limit:
            break
    if not lens:
        return {"n": 0}
    a = np.array(lens)
    return {
        "n": int(a.size),
        "median": int(np.median(a)),
        "p90": int(np.percentile(a, 90)),
        "p99": int(np.percentile(a, 99)),
        "max": int(a.max()),
        "share_ge_1024": round(float((a >= 1024).mean()), 4),
        "share_ge_4096": round(float((a >= 4096).mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3000, help="docs sampled per source")
    parser.add_argument("--out", type=str, default="/kaggle/working/doc_lengths.json")
    args = parser.parse_args()

    report: dict[str, dict] = {}
    for label, repo, config, field in SOURCES:
        print(f"\n=== {label}  ({repo}) ===", flush=True)
        try:
            stats = length_stats(iter_texts(repo, config, field, limit=args.limit), limit=args.limit)
            if stats.get("n", 0) == 0:
                # Never report an empty sample as a measurement. A zero here
                # means the column name was wrong, not that the source has no
                # documents -- and a false "no long documents" would be read as
                # evidence against the hypothesis.
                raise ValueError(
                    f"0 documents yielded from {repo} (field={field!r}); "
                    "treat as a probe failure, not a measurement"
                )
            report[label] = stats
            print(json.dumps(stats, indent=2), flush=True)
        except Exception as exc:  # noqa: BLE001 - a source may be unavailable
            report[label] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
            print(f"  unavailable: {type(exc).__name__}: {str(exc)[:160]}", flush=True)

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 64)
    print("What we need to move: the share of *training tokens* that sit inside a")
    print("document of at least context_len. At the current median of 239 tokens")
    print("almost no 1024-token window is a whole document.")
    ok = {k: v for k, v in report.items() if v.get("share_ge_1024", 0) >= 0.8}
    if ok:
        print(f"\nLong-form sources usable (>80% of docs >= 1024 tokens): {sorted(ok)}")
    else:
        print("\nNo source reached 80% long documents -- investigate before mixing.")


if __name__ == "__main__":
    main()
