"""The run registry shows gaps instead of hiding them.

The whole-record audit found the archive uneven: X17, X28 and X29 are fully
preserved while X16 — four arms, two seeds — has nothing at all, and nothing in
the repository made that visible. A registry that only lists what exists would
reproduce the problem it was built to fix, so the tests care most about the case
where a run has no artifacts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("build_runs_registry")


def test_the_committed_file_matches_the_archive():
    """A registry that drifts is worse than none: it would claim coverage the
    repository does not have."""
    assert (ROOT / "docs" / "results" / "RUNS.md").read_text() == R.build(ROOT)


def test_a_run_with_no_artifacts_is_reported_as_a_gap(tmp_path):
    """The X16 case, which is the reason this exists."""
    report = R.build(tmp_path)
    assert "**GAP** — nothing archived" in report
    assert "- **X16**" in report


def test_the_gap_list_names_every_missing_run(tmp_path):
    report = R.build(tmp_path)
    gaps = report.split("## Gaps")[1].split("## Artifacts")[0]
    for run_id in ("v0.1.0", "v0.1.1", "X16"):
        assert run_id in gaps
    # a run whose globs match nothing cannot be silently absent from the list
    assert gaps.count("- **") == len(R.RUNS)


def test_a_populated_run_is_counted_and_listed(tmp_path):
    target = tmp_path / "docs" / "results" / "entity-decode.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    report = R.build(tmp_path)
    assert "`entity decode` | Entity-consistent decoding vs greedy | — | 1 file |" in report
    assert "docs/results/entity-decode.json" in report


def test_the_totals_agree_with_the_rows(tmp_path):
    """A count that disagrees with its own table is the kind of quiet
    contradiction this project spent a day removing."""
    target = tmp_path / "docs" / "results" / "retrieval-probe.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    report = R.build(tmp_path)
    assert f"{len(R.RUNS)} runs tracked, 1 files archived" in report
    assert report.count("**GAP** — nothing archived") == len(R.RUNS) - 1


def test_every_declared_glob_actually_matches_something_or_is_a_known_gap():
    """Guards the registry against its own rot: a glob that has silently stopped
    matching would look like a gap forever, and a glob that never could match is
    a typo. Runs with no globs are the known-gap cases."""
    for run_id, _, _, globs in R.RUNS:
        if not globs:
            assert run_id in R.KNOWN_GAPS, f"{run_id} has no globs but no recorded reason"
            continue
        assert R.expand(globs, ROOT), f"{run_id}: globs match nothing -- typo or a moved archive"
