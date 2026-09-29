"""Tests for parsing `eval_val.py`'s output in the final-evaluation kernel.

`eval_val.py` writes **markdown**, not JSON. The first version of the
final-evaluation kernel assumed JSON, called `json.load` on a markdown file, and
died with a bare `JSONDecodeError` -- *after* the 100-batch eval had already run
to completion, and taking the benchmark half down with it. That is the same
failure shape as the x18-close runs: one assumption about a file format, no
assertion, whole kernel lost.

So the parser is pinned against the exact string `eval_val.py` builds, which is
constructed in its `main()` as::

    "# High-precision held-out loss\\n\\n"
    f"- Checkpoint: `{args.ckpt}`\\n"
    f"- Batches: {args.batches} x {args.batch_size} x {model.cfg.context_len} "
    f"= {tokens:,} tokens (seed {args.seed}, deterministic)\\n"
    f"- bytes/token: {bpt:.4f}\\n"
    f"- **Val loss: {mean_loss:.4f}**\\n"
    f"- **bpB: {bpb:.4f}**\\n"
"""

from __future__ import annotations

import re

SAMPLE = (
    "# High-precision held-out loss\n"
    "\n"
    "- Checkpoint: `/kaggle/working/ckpt.pt`\n"
    "- Batches: 100 x 8 x 1024 = 819,200 tokens (seed 1234, deterministic)\n"
    "- bytes/token: 4.4086\n"
    "- **Val loss: 2.8832**\n"
    "- **bpB: 0.9448**\n"
)


def parse(md: str) -> dict:
    """Mirror of the parsing cell in the final-evaluation notebook."""
    m_loss = re.search(r"Val loss:\s*([0-9.]+)", md)
    m_bpb = re.search(r"bpB:\s*([0-9.]+)", md)
    m_bpt = re.search(r"bytes/token:\s*([0-9.]+)", md)
    m_tok = re.search(r"=\s*([0-9,]+)\s*tokens", md)
    assert m_loss, f"could not parse val loss:\n{md}"
    assert m_bpb, f"could not parse bpB:\n{md}"
    return {
        "val_loss": float(m_loss.group(1)),
        "bpb": float(m_bpb.group(1)),
        "bytes_per_token": float(m_bpt.group(1)) if m_bpt else None,
        "tokens": m_tok.group(1) if m_tok else None,
    }


def test_parses_the_real_output_format():
    got = parse(SAMPLE)
    assert got["val_loss"] == 2.8832
    assert got["bpb"] == 0.9448
    assert got["bytes_per_token"] == 4.4086
    assert got["tokens"] == "819,200"


def test_verdict_thresholds_map_to_the_registered_bands():
    """P-A/B/C are decided by the number, so test the decision, not just the parse."""
    REF, GOOD, BAD = 3.0533, 3.0033, 3.1033

    def verdict(loss: float) -> str:
        if loss <= GOOD:
            return "P-B"
        if loss <= REF:
            return "noise"
        if loss <= BAD:
            return "worse"
        return "P-C"

    assert verdict(2.8832) == "P-B", "a 0.17 nat gain is a real improvement"
    assert verdict(3.0033) == "P-B", "the band edge is inclusive"
    assert verdict(3.0533) == "noise", "exactly at the reference is matched"
    assert verdict(3.08) == "worse", "between REF and BAD is matched-within-noise"
    assert verdict(3.1034) == "P-C", "a real regression is reported as one"


def test_parsing_markdown_that_is_actually_json_is_still_rejected_loudly():
    """The original bug, as a test: a JSON body must not silently parse as 0.0."""
    try:
        parse('{"val_loss": 2.88}')
    except AssertionError:
        return
    raise AssertionError("expected an AssertionError, not a silently wrong number")
