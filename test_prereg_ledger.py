"""The ledger must be checked by the suite, not only by hand.

The hash-anchored pre-registration ledger is the project's strongest integrity
claim, and `verify_prereg_ledger.py` is the tool that backs it. It verifies 21
entries at their registration commits and classifies tree drift as extension
(insertions only) or in-place edit, so a reader knows which files' current text
is not the registered text.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def test_ledger_verifier_passes_on_the_committed_ledger():
    r = subprocess.run(
        [sys.executable, "tools/verify_prereg_ledger.py"],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "entries verify" in r.stdout
    assert "FAILED" not in r.stdout


def test_drift_classification_separates_insertion_from_rewrite():
    from tools.verify_prereg_ledger import classify_drift

    registered = ["# X", "prediction: it works", "results: pending"]
    extended = registered + ["results: see below"]
    label, inserted, deleted = classify_drift(registered, extended)
    assert (label, inserted, deleted) == ("extended", 1, 0)

    rewritten = ["# X", "prediction: it worked all along", "results: pending"]
    label, _, deleted = classify_drift(registered, rewritten)
    assert label == "EDITED", "a rewritten prediction must not be labelled appended"
    assert deleted == 1

    shortened = registered[:-1]
    label, _, deleted = classify_drift(registered, shortened)
    assert label == "EDITED", "deleting registered text must not be labelled appended"
    assert deleted == 1
