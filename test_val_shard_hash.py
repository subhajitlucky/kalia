"""Pin the canonical yardstick shards by hash (G2 of the pre-launch gates).

D43: a corpus rebuild silently redefined the val set and produced a 0.74-nat
"regression" that was entirely an artifact of comparing across sets. The fix is
mechanical: the sha256 of every shard a measurement reads is recorded next to
the measurement, and this test fails if the shard changes without the record
changing with it.

The hashes live in `docs/legal/val-shard-hashes.json`, written from the
artifacts Step 1 actually reads (G0) — not from prep outputs. Until that file
exists, these tests skip with an explicit reason instead of passing silently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
HASH_FILE = ROOT / "docs" / "legal" / "val-shard-hashes.json"

# Shards whose identity defines the v0.3.0 yardstick. Paths are relative to
# the dataset/model repo root as seen by the Kaggle kernels.
PINNED_SHARDS = ("corpus/probe_val.bin",)


def _recorded_hashes() -> dict:
    if not HASH_FILE.exists():
        pytest.skip(f"{HASH_FILE.name} not recorded yet (G2 open) — record it from G0 artifacts")
    record = json.loads(HASH_FILE.read_text())
    assert isinstance(record, dict) and record, f"{HASH_FILE} must map shard path to sha256"
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_hash_record_covers_every_pinned_shard():
    recorded = _recorded_hashes()
    missing = [shard for shard in PINNED_SHARDS if shard not in recorded]
    assert not missing, f"no recorded hash for pinned shards: {missing}"


def test_recorded_hashes_are_well_formed():
    recorded = _recorded_hashes()
    for shard, value in recorded.items():
        assert re_full_match(value), f"{shard}: not a 64-char hex sha256: {value!r}"


def re_full_match(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def test_local_shards_match_the_record():
    """Runs wherever the shards are present; skips file-by-file otherwise."""
    recorded = _recorded_hashes()
    checked = 0
    for shard, expected in recorded.items():
        for root in (ROOT, Path("/kaggle/input")):
            candidates = list(root.rglob(shard)) if root.exists() else []
            for path in candidates:
                assert _sha256(path) == expected, f"{path} does not match the recorded hash"
                checked += 1
    if checked == 0:
        pytest.skip("no pinned shards present locally — hashes verified at record time")
