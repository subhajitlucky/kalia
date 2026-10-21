"""Tests for X31's archival verdict tool.

The bars are fixed in the pre-registration, so the tool that decides pass/fail
must be tested against synthetic curves with known outcomes. A verdict tool that
silently reads a missing arm as zero, or averages one seed, would be the same
class of failure the experiment exists to avoid.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.x31_verdict import compute

SEEDS = ("1338", "1339")


def _write(directory: Path, condition: str, seed: str, values: list[float]) -> None:
    lines = ["step,val_loss"] + [f"{(i + 1) * 100},{v}" for i, v in enumerate(values)]
    (directory / f"val_log-{condition}-s{seed}.csv").write_text("\n".join(lines) + "\n")


def _cohort(directory: Path, deltas: dict[str, dict[str, float]]) -> None:
    """Control is 4.80 at both seeds; each condition is control + its delta."""
    for seed in SEEDS:
        _write(directory, "control", seed, [5.10, 4.80])
        for condition in ("qk", "cap", "both"):
            _write(directory, condition, seed, [5.10, 4.80 + deltas[condition][seed]])


def test_bundle_that_replicates_and_is_carried_by_qk(tmp_path):
    _cohort(tmp_path, {
        "qk": {"1338": -0.080, "1339": -0.070},
        "cap": {"1338": -0.001, "1339": +0.002},
        "both": {"1338": -0.085, "1339": -0.072},
    })
    v = compute(tmp_path)
    assert v["q1"] and v["q1b"]
    assert v["q2a"] and not v["q2b"]
    assert v["q3"]
    assert v["decision"] == "bundle replicated at two seeds"


def test_one_seed_carrying_the_bundle_is_not_replication(tmp_path):
    _cohort(tmp_path, {
        "qk": {"1338": -0.090, "1339": +0.005},
        "cap": {"1338": -0.002, "1339": -0.001},
        "both": {"1338": -0.100, "1339": +0.004},
    })
    v = compute(tmp_path)
    assert v["q1"], "the mean clears the bar"
    assert not v["q1b"], "but the second seed is positive"
    assert v["decision"].startswith("unreplicated")


def test_a_bundle_that_does_not_replicate_is_withdrawn(tmp_path):
    _cohort(tmp_path, {
        "qk": {"1338": +0.010, "1339": -0.005},
        "cap": {"1338": +0.002, "1339": +0.001},
        "both": {"1338": +0.012, "1339": -0.004},
    })
    v = compute(tmp_path)
    assert not v["q1"]
    assert v["decision"].startswith("not replicated")


def test_missing_arm_is_an_error_not_a_zero(tmp_path):
    _cohort(tmp_path, {
        "qk": {"1338": -0.080, "1339": -0.070},
        "cap": {"1338": 0.0, "1339": 0.0},
        "both": {"1338": -0.085, "1339": -0.072},
    })
    (tmp_path / "val_log-cap-s1339.csv").unlink()
    with pytest.raises(FileNotFoundError):
        compute(tmp_path)
