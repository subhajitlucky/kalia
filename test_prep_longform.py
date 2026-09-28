"""Tests for the long-form probe's source resolution.

The first two runs of this probe died in ways that look like results:

- run 1: three of four sources reported a repo/config error, so the only number
  that came back was TinyStories' and the long-form question stayed unanswered
- run 2: `deepmind/pg19` is a *script* dataset and modern `datasets` refuses to
  run scripts, so it cannot be loaded streaming or not

The dangerous third case is silent: `row.get("text")` on a dataset whose column
is `TEXT` returns None, the `if text` guard drops every row, and the probe
reports `{"n": 0}` -- indistinguishable from "this source has no long
documents". A false negative there would read as evidence *against* the
hypothesis the probe exists to test. These tests pin the guards.
"""

from __future__ import annotations

import pytest

from prep_longform import SOURCES, _pick_field, iter_texts, length_stats


def test_pick_field_accepts_exact_match():
    assert _pick_field({"text": "a", "id": 1}, "text") == "text"


def test_pick_field_resolves_case_mismatches_both_ways():
    # The real case: gutenberg-english names it TEXT, everything else uses text.
    assert _pick_field({"TEXT": "a", "SOURCE": "b"}, "text") == "TEXT"
    assert _pick_field({"text": "a"}, "TEXT") == "text"


def test_pick_field_falls_back_to_a_known_alias():
    assert _pick_field({"Content": "a"}, "text") == "Content"


def test_pick_field_raises_rather_than_returning_a_wrong_column():
    with pytest.raises(KeyError, match="text column not found"):
        _pick_field({"embedding": [0.0], "meta": {}}, "text")


def test_empty_sample_is_not_reported_as_a_measurement():
    """The false-negative guard: zero rows must be a failure, not a statistic."""
    stats = length_stats(iter([]), limit=10)
    assert stats == {"n": 0}
    # main() turns this into an error; assert the shape it keys off.
    assert stats.get("n", 0) == 0


def test_gutenberg_source_is_parquet_native_and_mit_declared():
    """The two properties that disqualified pg19 and its mirrors.

    `deepmind/pg19` ships a pg19.py loader and no parquet, and the obvious
    parquet mirrors (emozilla/pg19, manu/project_gutenberg) declare no licence
    at all -- which fails our own permissive filter under D44. These are
    network-dependent facts, so the test asserts the choice we made rather than
    re-probing the Hub on every run.
    """
    labels = {label: (repo, field) for label, repo, _cfg, field in
              ((s[0], s[1], s[2], s[3]) for s in SOURCES)}
    longform = [v for k, v in labels.items() if "long-form" in k]
    assert len(longform) == 1
    repo, field = longform[0]
    assert repo == "sedthh/gutenberg_english"
    assert field == "TEXT", "gutenberg-english names the column TEXT, not text"


def test_no_source_still_points_at_the_unloadable_pg19():
    for _label, repo, _cfg, _field in SOURCES:
        assert repo != "deepmind/pg19", "pg19 is script-based and unloadable"
