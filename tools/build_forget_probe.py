"""Split the canonical val set into a decision half and a frozen forgetting probe.

Why: the canonical `val.bin` is currently used for *decisions* -- the in-training
eval and the J2 stopping rule. A model that is repeatedly selected against a set
can look stable on that set while quietly eroding elsewhere, so we cannot detect
forgetting by measuring on it.

The fix is to partition it once, by document, into:
  - `val_regression.bin` : what the training loop watches (today's behaviour)
  - `val_forget.bin`     : frozen, never used for any decision, only for the
                           forgetting curve

Both halves keep the same 4-corpus mixture, because the mixture repeats every
1M-token round in the mixer (`mix_bins.py: block_tokens=1_000_000`), so a
contiguous split at a round boundary is corpus-balanced by construction.

The probe is the instrument that makes continual learning measurable at all: it is
the fixed yardstick that CL-1 (replay), CL-2 (merging) and CL-4 (SVD-constrained
updates) are all tested against. Built once, never regenerated.

Usage:
    python tools/build_forget_probe.py --val-bin data/val.bin --out-dir data/probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DOC_SEPARATOR = 50256


def split_at_document(tokens: np.ndarray, fraction: float = 0.5) -> int:
    """Return an index that is a document boundary, nearest to `fraction` of the file.

    Walks EOS positions rather than cutting at an arbitrary token, so neither half
    begins or ends mid-document.
    """
    separators = np.flatnonzero(tokens == DOC_SEPARATOR)
    if separators.size == 0:
        raise ValueError("no document separators found; cannot split safely")
    target = int(len(tokens) * fraction)
    idx = int(np.searchsorted(separators, target))
    idx = min(max(idx, 1), len(separators) - 1)
    # Cut *after* the separator so document i stays in the first half.
    return int(separators[idx]) + 1


def describe(tokens: np.ndarray) -> dict:
    separators = np.flatnonzero(tokens == DOC_SEPARATOR)
    lengths = np.diff(separators)
    lengths = lengths[lengths > 0]
    return {
        "tokens": int(len(tokens)),
        "documents": int(len(separators)),
        "starts_with_sep": bool(tokens[0] == DOC_SEPARATOR),
        "ends_with_sep": bool(tokens[-1] == DOC_SEPARATOR),
        "median_doc_len": int(np.median(lengths)) if lengths.size else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-bin", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    args = parser.parse_args()

    tokens = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    cut = split_at_document(np.asarray(tokens[:2_000_000]), args.fraction)
    # Recompute the cut over the whole file, still on a separator.
    sep_all = np.flatnonzero(np.asarray(tokens) == DOC_SEPARATOR)
    target = int(len(tokens) * args.fraction)
    j = int(np.searchsorted(sep_all, target))
    j = min(max(j, 1), len(sep_all) - 1)
    cut = int(sep_all[j]) + 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    head = np.asarray(tokens[:cut], dtype=np.uint16)
    tail = np.asarray(tokens[cut:], dtype=np.uint16)
    # Trim each half so it both starts and ends on a document boundary.
    first_tail_sep = np.flatnonzero(tail == DOC_SEPARATOR)
    if first_tail_sep.size:
        tail = tail[first_tail_sep[0] :]
    tail = np.concatenate([tail, np.array([DOC_SEPARATOR], dtype=np.uint16)])
    if head.size and head[-1] != DOC_SEPARATOR:
        head = head[:-1]
    if head.size and head[0] == DOC_SEPARATOR:
        head = head[1:]

    reg = args.out_dir / "val_regression.bin"
    fog = args.out_dir / "val_forget.bin"
    head.tofile(reg)
    tail.tofile(fog)

    meta = {
        "source": str(args.val_bin),
        "cut_index": cut,
        "regression": {str(reg): describe(head)},
        "forget": {str(fog): describe(tail)},
    }
    (args.out_dir / "probe_manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(
        "\nThe forgetting probe is frozen from this point. Regenerating it resets the"
        "\nforgetting curve, so only do that deliberately."
    )


if __name__ == "__main__":
    main()
