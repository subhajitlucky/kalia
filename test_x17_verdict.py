"""X17's verdict is computed, not transcribed.

The registration fixes five bars and the decision rule. X21's verdict was
assembled by hand, which is one transcription away from being wrong. These tests
pin the computation on synthetic archives, including the cases that look like a
pass and are not — a mean met by one seed while the other moves the wrong way,
and a treatment that wins by being slower.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent


def _load(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load("x17_verdict")

STEPS = (100, 200, 300, 400, 500)


def _archive(tmp_path: Path, *, control: dict, treatment: dict, tps: dict,
             engagement: float = 0.42) -> Path:
    """Write a synthetic archive in the shape the run leaves behind."""
    for seed in V.SEEDS:
        for arm, curve in (("control", control), ("treatment", treatment)):
            rows = ["step,val_loss"] + [f"{s},{curve[seed][s]}" for s in STEPS]
            (tmp_path / f"val_log-{arm}-s{seed}.csv").write_text("\n".join(rows) + "\n")

    lines = []
    for seed in V.SEEDS:
        for arm in ("control", "treatment"):
            lines.append(f"ARM: {arm}-s{seed}")
            rate = f"{tps[(arm, seed)]:,}"
            lines += [f"step {s}/500 | loss 3.0000 | lr 6.00e-04 | {rate} tok/s"
                      for s in STEPS]
    (tmp_path / "kalia-x17-docmask.log").write_text("\n".join(lines) + "\n")

    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"dm4_mask_engagement": engagement})
    )
    return tmp_path


def _flat(value: float) -> dict:
    return {seed: {s: value for s in STEPS} for seed in V.SEEDS}


def _curve(s1401: float, s1402: float) -> dict:
    return dict(zip(V.SEEDS, ({s: v for s in STEPS} for v in (s1401, s1402))))


def _passing(tmp_path: Path) -> Path:
    return _archive(
        tmp_path,
        control=_curve(3.9000, 3.9100),
        treatment=_curve(3.8700, 3.8800),   # deltas -0.030 both seeds
        tps={("control", "1338"): 30000, ("treatment", "1338"): 28000,
             ("control", "1339"): 30000, ("treatment", "1339"): 28500},
    )


def test_a_clean_pass_promotes(tmp_path):
    v = V.compute(_passing(tmp_path))
    assert v["dm1"] and v["dm1b"] and v["dm2"] and v["dm3"] and v["dm4"]
    assert v["promote"] is True
    assert v["dm1_mean_delta"] == pytest.approx(-0.030)
    assert v["dm5_spread"] == pytest.approx(0.0)


def test_a_mean_carried_by_one_seed_fails_dm1b(tmp_path):
    """The case X21's R-2 exists to refuse: the mean clears the bar while one
    seed moves the wrong way."""
    root = _archive(
        tmp_path,
        control=_curve(3.9000, 3.9100),
        treatment=_curve(3.8400, 3.9200),   # -0.060 then +0.010
        tps={("control", s): 30000 for s in V.SEEDS} | {("treatment", s): 29000 for s in V.SEEDS},
    )
    v = V.compute(root)
    assert v["dm1_mean_delta"] == pytest.approx(-0.025)
    assert v["dm1"] is True, "the mean does clear the bar"
    assert v["dm1b"] is False, "but one seed moved the wrong way"
    assert v["promote"] is False


def test_missing_the_bar_fails_dm1(tmp_path):
    root = _archive(
        tmp_path,
        control=_curve(3.9000, 3.9100),
        treatment=_curve(3.8950, 3.9050),   # -0.005 both seeds: concordant, too small
        tps={("control", s): 30000 for s in V.SEEDS} | {("treatment", s): 29500 for s in V.SEEDS},
    )
    v = V.compute(root)
    assert v["dm1"] is False
    assert v["dm1b"] is True, "the direction is right; the size is not"
    assert v["promote"] is False


def test_winning_by_being_slower_fails_dm3(tmp_path):
    """A correctness fix that costs 40% throughput is a different decision, and
    DM-3 is the bar that says so."""
    root = _archive(
        tmp_path,
        control=_curve(3.9000, 3.9100),
        treatment=_curve(3.8700, 3.8800),
        tps={("control", s): 30000 for s in V.SEEDS} | {("treatment", s): 21000 for s in V.SEEDS},
    )
    v = V.compute(root)
    assert v["dm1"] is True
    assert v["dm3"] is False
    assert all(r == pytest.approx(30000 / 21000) for r in v["dm3_ratios"].values())


def test_an_inert_mask_fails_dm4(tmp_path):
    """Engagement exactly 1.0 means every causally-eligible position was
    attended: the mask did nothing and the treatment duplicates the control."""
    root = _archive(
        tmp_path,
        control=_flat(3.9000),
        treatment=_flat(3.8800),
        tps={("control", s): 30000 for s in V.SEEDS} | {("treatment", s): 29500 for s in V.SEEDS},
        engagement=1.0,
    )
    v = V.compute(root)
    assert v["dm4"] is False


def test_an_effect_that_only_appears_at_the_end_fails_dm2(tmp_path):
    """DM-2 asks for the effect at >=3 of 5 eval points, not just the last one:
    a single late crossing is as likely to be noise as a mechanism. Here the
    final delta is large enough to clear DM-1 comfortably, and DM-2 still
    refuses it."""
    control = _curve(3.9000, 3.9100)
    treatment = {seed: {s: (3.9100 if s < 500 else 3.8000) for s in STEPS} for seed in V.SEEDS}
    root = _archive(
        tmp_path,
        control=control,
        treatment=treatment,
        tps={("control", s): 30000 for s in V.SEEDS} | {("treatment", s): 29500 for s in V.SEEDS},
    )
    v = V.compute(root)
    assert v["dm1"] is True, "the final number clears the bar"
    assert v["dm1b"] is True, "and both seeds agree"
    assert v["dm2"] is False, "but it is ahead at one point of five"
    assert v["promote"] is False


def test_a_missing_archive_says_so_instead_of_guessing(tmp_path):
    with pytest.raises(FileNotFoundError, match="has the run been archived"):
        V.compute(tmp_path)


def test_a_missing_log_is_reported_not_defaulted(tmp_path):
    root = _passing(tmp_path)
    (root / "kalia-x17-docmask.log").unlink()
    with pytest.raises(FileNotFoundError, match="DM-3 needs its throughput"):
        V.compute(root)


def test_the_exit_code_encodes_computability_not_outcome(tmp_path, capsys):
    """A failed experiment must not look like a broken build."""
    import subprocess
    import sys

    root = _archive(
        tmp_path,
        control=_curve(3.9000, 3.9100),
        treatment=_curve(3.8950, 3.9050),
        tps={("control", s): 30000 for s in V.SEEDS} | {("treatment", s): 29500 for s in V.SEEDS},
    )
    r = subprocess.run(
        [sys.executable, "tools/x17_verdict.py", "--dir", str(root)],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "do not promote" in r.stdout

    missing = subprocess.run(
        [sys.executable, "tools/x17_verdict.py", "--dir", str(tmp_path / "nope")],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert missing.returncode == 2
