"""The claims checker, falsified against the defects it exists to prevent.

A checker that only ever passes on the current tree proves nothing: it may be
matching nothing at all. Each test here re-introduces one of the nine defects the
2026-10-18 audit found, and asserts the rule fires. `test_the_live_tree_passes`
is the control -- the mutations below are the only difference, so a rule that
fired on everything would fail it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent


def _load(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load("audit_claims")


def _tree(tmp_path: Path) -> Path:
    """A copy of every file the audit reads, so a test can damage one."""
    for rel in [*audit.PUBLIC, audit.DECISIONS, audit.GATE]:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text((ROOT / rel).read_text())
    return tmp_path


def _replace(root: Path, rel: str, old: str, new: str) -> None:
    path = root / rel
    text = path.read_text()
    assert old in text, f"{rel}: anchor not found; the document moved under this test"
    path.write_text(text.replace(old, new, 1))


def _replace_all(root: Path, rel: str, old: str, new: str) -> None:
    path = root / rel
    text = path.read_text()
    assert old in text, f"{rel}: anchor not found; the document moved under this test"
    path.write_text(text.replace(old, new))


def test_the_live_tree_passes():
    assert audit.audit(ROOT) == {name: [] for name in audit.CHECKS}


def test_a_card_omitting_the_code_corpus_is_caught(tmp_path):
    """The defect: both cards' HF YAML listed three sources and dropped the
    code corpus, which is 5% of the training mixture."""
    root = _tree(tmp_path)
    _replace(root, audit.CARDS[1], "- codeparrot/codeparrot-clean\n", "")
    problems = audit.check_datasets(root)
    assert problems and "codeparrot/codeparrot-clean" in problems[0]


def test_a_card_calling_the_probe_held_out_is_caught(tmp_path):
    """The defect, twice: the shard notice on 2026-10-18 and then the v0.2.0
    card, which correcting the notice missed."""
    root = _tree(tmp_path)
    _replace(
        root,
        audit.CARDS[1],
        "deliberately not held-out data, and not the canonical validation set",
        "held-out windows from the sources above",
    )
    problems = audit.check_probe_role(root)
    assert problems and "held-out" in problems[0]


def test_a_citation_of_a_decision_that_does_not_exist_is_caught(tmp_path):
    """The defect: an index pointing at a table that had stopped six entries
    earlier, so the newest decision a reader could find was the wrong one."""
    root = _tree(tmp_path)
    _replace(root, "README.md", "zero budget", "zero budget (see D999)")
    problems = audit.check_decision_refs(root)
    assert problems and "D999" in problems[0]


def test_a_release_number_drifting_from_the_artifact_is_caught(tmp_path):
    """No defect yet. This is the rule that keeps the one number the release
    rests on attached to the file that was actually re-scored. Every occurrence
    is changed: the README states it in two places, and a check satisfied by
    either one would not be checking the claim."""
    root = _tree(tmp_path)
    _replace_all(root, "README.md", "2.8248", "2.8250")
    problems = audit.check_release_numbers(root)
    assert problems and "2.8248" in problems[0]


def test_a_notice_whose_table_shape_moved_is_caught(tmp_path):
    """The failure mode of the checker itself: if the notice stops declaring
    shard rows, `required` goes empty and every card trivially satisfies it.
    That has to be a failure, not a pass."""
    root = _tree(tmp_path)
    (root / audit.NOTICE).write_text("# Corpus shards\n\nProse only, no table.\n")
    problems = audit.check_datasets(root)
    assert problems and "parsed no source datasets" in problems[0]


def test_a_card_without_front_matter_is_an_error_not_a_pass(tmp_path):
    """A card that moved its YAML would otherwise be read as attributing
    nothing, which is vacuously fine -- so it must raise."""
    root = _tree(tmp_path)
    (root / audit.CARDS[1]).write_text("# no front matter here\n")
    with pytest.raises(ValueError, match="front matter"):
        audit.card_datasets(root, audit.CARDS[1])
