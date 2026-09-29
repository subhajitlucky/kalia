"""Tests for docs/results/experiments.json — the run-result record.

The X21 result that closed the architecture line was, until this file existed,
in exactly two places: a markdown table in a pre-registration, and a Kaggle log
that is transient and partially truncated. The four X21 validation curves existed
nowhere at all. So the questions these tests ask are not academic:

- Does every committed number have a source, and is that source honest about
  whether it was parsed from a run or written down by a human?
- Do the deltas in the file agree with the deltas in the pre-registrations, or
  has prose drifted from data?
- Does the archive still regenerate identically, or has it gone stale?
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "docs" / "results" / "experiments.json").read_text())
BY_EXP = {e["experiment"]: e for e in DATA["experiments"]}

# The single-seed lineage. These are the numbers D48 reclassified, so they are
# the ones most likely to be quietly "corrected" later.
LEGACY = {
    "x18-close": {"micro-base": 4.7662, "micro-gated": 4.7226, "micro-nope": 4.7772},
    "x19-branchnorm": {"micro-base": 4.7662, "micro-branchnorm": 4.7680},
    "x20-staticgate": {"micro-base": 4.7662, "micro-staticgate": 4.6304},
}
X21_FINALS = {
    "micro-base-s1338": 4.8769,
    "micro-staticgate-s1338": 4.9623,
    "micro-base-s1339": 4.8908,
    "micro-staticgate-s1339": 4.9087,
}


def arms(exp: str) -> dict:
    return {a["arm"]: a for a in BY_EXP[exp].get("arms", [])}


def test_every_experiment_that_ran_is_recorded():
    for name in ("x18-close", "x19-branchnorm", "x20-staticgate", "x21"):
        assert name in BY_EXP, f"{name} is missing from the results record"
        assert BY_EXP[name].get("present"), f"{name} has no archived log"
        assert BY_EXP[name]["kernel"].startswith("subhajitlucky/"), name


def test_every_arm_carries_a_source_and_the_source_is_classified():
    for exp, entry in BY_EXP.items():
        for a in entry.get("arms", []):
            assert a.get("source"), f"{exp}/{a['arm']} has no source"
            if a["source"].startswith("prereg:"):
                # Prose-transcribed. Must not pretend to be a parsed log, and
                # must be flagged, because these are the weakest records here.
                assert a.get("truncated") is True, f"{exp}/{a['arm']} needs truncated=True"
                assert a.get("val_curve") is None
            else:
                assert Path(ROOT / a["source"]).exists(), f"{exp}/{a['arm']} cites a missing log"


def test_no_archived_log_is_claimed_to_hold_a_curve_it_lacks():
    """A curve must be backed by the log, not by the value it ends at."""
    for exp, entry in BY_EXP.items():
        for a in entry.get("arms", []):
            if a.get("val_curve") is not None:
                assert not a["source"].startswith("prereg:"), (
                    f"{exp}/{a['arm']}: a prereg-only record cannot carry a curve"
                )
                assert max(int(k) for k in a["val_curve"]) == 500
                assert a["val_curve"][str(500)] == a["val_loss_final"], (
                    f"{exp}/{a['arm']}: final value disagrees with its own curve"
                )


def test_legacy_single_seed_numbers_match_their_preregistrations():
    """Prose and data must not drift apart.

    These are the numbers D48 turned into nulls. If someone edits a markdown
    table to make a result look better, this fails rather than letting the two
    versions disagree quietly.
    """
    for exp, expected in LEGACY.items():
        got = arms(exp)
        for arm, val in expected.items():
            assert arm in got, f"{exp}: {arm} missing"
            assert got[arm]["val_loss_final"] == val, f"{exp}/{arm} drifted from its prereg"


def test_prereg_deltas_still_say_what_the_markdown_says():
    md = (ROOT / "docs" / "preregistrations" / "2026-09-29-X20-static-gate.md").read_text()
    got = arms("x20-staticgate")
    # The -0.1358 that the model card would otherwise have claimed.
    assert got["micro-staticgate"]["delta_vs_control"] == -0.1358
    assert "−0.1358" in md or "-0.1358" in md
    # And X18's -0.0436, which D48 reclassified as a null.
    assert round(arms("x18-close")["micro-gated"]["delta_vs_control"], 4) == -0.0436


def test_x21_records_all_four_arms_with_their_finals():
    got = arms("x21")
    assert set(got) == set(X21_FINALS), f"X21 arms drifted: {sorted(got)}"
    for arm, val in X21_FINALS.items():
        assert got[arm]["val_loss_final"] == val


def test_x21_deltas_reproduce_the_registered_verdict():
    """R-1 and R-2 recomputed from the data, not read from prose.

    If the file is the source of truth, the verdict must fall out of it.
    """
    got = arms("x21")
    pairs = []
    for seed in (1338, 1339):
        c, t = got[f"micro-base-s{seed}"], got[f"micro-staticgate-s{seed}"]
        pairs.append(round(t["val_loss_final"] - c["val_loss_final"], 4))
    assert pairs == [0.0854, 0.0179], pairs
    mean = sum(pairs) / 2
    # The kernel printed 0.0516, computed from full-precision floats. We store
    # 4-dp losses, and the rounded deltas average to 0.0517 instead. Asserting
    # the exact printed digit would pin a number this file cannot reproduce, so
    # the tolerance covers the rounding and nothing more.
    assert 0.0515 < mean < 0.0518, mean
    # R-1 needed mean <= -0.050; R-2 needed both deltas <= -0.050.
    assert not (mean <= -0.050)
    assert not all(d <= -0.050 for d in pairs)
    assert mean > 0, "the effect inverted; a future edit must not hide that"


def test_the_control_baseline_spread_is_recorded_not_just_asserted():
    """D48's actual finding: the control is not seed-invariant.

    If this test ever needs its expectation changed, that change is the claim
    that a 0.11-nat spread was noise, and deserves a decision entry.
    """
    got = arms("x21")
    fresh = [got[f"micro-base-s{s}"]["val_loss_final"] for s in (1338, 1339)]
    legacy = LEGACY["x18-close"]["micro-base"]
    spread = min(fresh) - legacy
    assert 0.09 < spread < 0.13, f"control spread moved to {spread:.4f}"
    # Larger than every effect D48 reclassified as noise.
    assert spread > 0.0436
    assert spread > 0.0018


def test_archived_logs_regenerate_the_committed_file():
    """Guards against a hand-edit to experiments.json.

    The file is generated. If it is also hand-maintained, the two drift and
    nobody notices, which is the failure this whole exercise is about.
    """
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "tools/collect_results.py", "--check"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_parameter_counts_are_present_and_ordered_sensibly():
    got = arms("x20-staticgate")
    base, static = got["micro-base"]["params"], got["micro-staticgate"]["params"]
    assert base == 29_920_512
    assert static == 29_999_040
    # X20's H-1 needed it to be *smaller* than the data-dependent gate.
    assert static < arms("x18-close")["micro-gated"]["params"] == 30_072_768


def test_record_makes_its_own_weaknesses_findable():
    note = DATA["note"].lower()
    assert "truncat" in note
    assert any(
        a.get("truncated") for e in DATA["experiments"] for a in e.get("arms", [])
    ), "if nothing is truncated, the archive got better and this test should be revisited"
