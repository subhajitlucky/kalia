"""Tokenize text sources into uint16 .bin shards for KALIA training.

Run this on a Kaggle CPU notebook (CPU quota is unlimited) or locally for smoke tests.
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

WRITE_CHUNK = 1_000_000  # tokens buffered before flushing to disk


def iter_tinystories() -> Iterator[dict]:
    from datasets import load_dataset

    yield from load_dataset("roneneldan/TinyStories", split="train", streaming=True)


def iter_fineweb() -> Iterator[dict]:
    from datasets import load_dataset

    yield from load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True
    )


def iter_smollm_fineweb_edu() -> Iterator[dict]:
    """Deduplicated FineWeb-Edu from the SmolLM corpus (curated for small models)."""
    from datasets import load_dataset

    for row in load_dataset(
        "HuggingFaceTB/smollm-corpus", name="fineweb-edu-dedup", split="train", streaming=True
    ):
        yield {"text": row["text"]}


def iter_cosmopedia() -> Iterator[dict]:
    """Synthetic educational textbooks/blog posts/stories from Cosmopedia v2."""
    from datasets import load_dataset

    for row in load_dataset(
        "HuggingFaceTB/smollm-corpus", name="cosmopedia-v2", split="train", streaming=True
    ):
        yield {"text": row["text"]}


PERMISSIVE_CODE_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "bsd-2-clause",
    "isc",
    "unlicense",
    "cc0-1.0",
}


def is_permissive_license(license_id: str | None) -> bool:
    """True when a code file's license allows redistribution without copyleft."""
    return (license_id or "").strip().lower() in PERMISSIVE_CODE_LICENSES


def iter_stack_smol() -> Iterator[dict]:
    """Python code filtered to permissive licenses (clean for public release)."""
    from datasets import load_dataset

    for row in load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True):
        if is_permissive_license(row.get("license")):
            yield {"text": row["content"]}


SOURCES: dict[str, Callable[[], Iterable[dict]]] = {
    "tinystories": iter_tinystories,
    "fineweb": iter_fineweb,
    "smollm_fineweb_edu": iter_smollm_fineweb_edu,
    "cosmopedia": iter_cosmopedia,
    "stack_smol": iter_stack_smol,
}


def tokenize_documents(
    docs: Iterable[dict],
    encoder,
    out_path: str | Path,
    max_tokens: int,
) -> int:
    """Tokenize documents to a uint16 file. Returns tokens written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    buffer: list[int] = []

    def flush(fh) -> None:
        nonlocal buffer, written
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(fh)
            written += len(buffer)
            buffer = []

    with open(out_path, "wb") as fh:
        for doc in docs:
            ids = encoder.encode_ordinary(doc["text"])
            ids.append(encoder.eot_token)
            buffer.extend(ids)
            if len(buffer) >= WRITE_CHUNK:
                flush(fh)
            if written + len(buffer) >= max_tokens:
                break
        flush(fh)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize a source into a .bin shard")
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--out", type=Path, required=True, help="output .bin path")
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--meta", type=Path, default=None, help="optional meta.json path")
    args = parser.parse_args()

    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    docs = SOURCES[args.source]()
    n = tokenize_documents(docs, encoder, args.out, args.max_tokens)
    print(f"wrote {n} tokens to {args.out}")

    if args.meta is not None:
        meta = {}
        if args.meta.exists():
            meta = json.loads(args.meta.read_text())
        meta[args.out.name] = {"source": args.source, "tokens": n}
        args.meta.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
