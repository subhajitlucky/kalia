"""The 2x2 decomposition, on synthetic cells.

X17's recorded penalty is real *as measured* and reverses under matched
evaluation. The arithmetic that says so has to be right, and the cases that
matter are the ones where the two rows disagree in sign — which is exactly what
happened, so a tool that assumed one direction would have called it agreement.
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


M = _load("x17_mask2x2")


def _manifest(cells: dict) -> dict:
    return {"cells": cells}


def _cells(control_unmasked, control_masked, treat_unmasked, treat_masked) -> dict:
    """Four per-seed values; keys exactly as the eval kernel writes them."""
    out = {}
    for i, seed in enumerate(M.SEEDS):
        out[f"control-s{seed}|eval_unmasked"] = control_unmasked[i]
        out[f"control-s{seed}|eval_masked"] = control_masked[i]
        out[f"treatment-s{seed}|eval_unmasked"] = treat_unmasked[i]
        out[f"treatment-s{seed}|eval_masked"] = treat_masked[i]
    return out


def test_the_real_shape_reproduces_the_recorded_penalty():
    """The numbers the run actually produced: +0.0493/+0.0400 mismatched, which
    is X17's recorded +0.0444 reproduced under fixed windows, and -0.0088/-0.0160
    matched."""
    v = M.compute(_manifest(_cells(
        [4.809422, 4.798057], [4.818221, 4.806189],
        [4.858676, 4.838087], [4.809400, 4.790205],
    )))
    assert v["deltas"]["s1338|unmasked"] == pytest.approx(0.049254, abs=1e-6)
    assert v["deltas"]["s1339|masked"] == pytest.approx(-0.015984, abs=1e-6)
    assert v["mean_mismatched"] == pytest.approx(0.044642, abs=1e-6)
    assert v["mean_matched"] == pytest.approx(-0.012403, abs=1e-6)
    assert v["interaction"] == pytest.approx(-0.057045, abs=1e-6)
    assert v["control_eval_mask_shift"] == pytest.approx(0.008465, abs=1e-6)


def test_both_rows_concordant_and_pointing_opposite_ways():
    """Concordance is agreement in sign, not a direction. A checker that asked
    "are both negative" would call the mismatched row discordant when it is in
    fact perfectly consistent -- in the other direction."""
    v = M.compute(_manifest(_cells(
        [4.80, 4.80], [4.81, 4.81], [4.85, 4.84], [4.79, 4.79],
    )))
    assert v["mean_mismatched"] > 0 and v["mismatched_concordant"] is True
    assert v["mean_matched"] < 0 and v["matched_concordant"] is True


def test_a_genuinely_discordant_row_is_reported_as_such():
    v = M.compute(_manifest(_cells(
        [4.80, 4.80], [4.81, 4.81], [4.85, 4.76], [4.79, 4.79],
    )))
    assert v["mismatched_concordant"] is False


def test_the_interaction_is_the_difference_of_the_two_means():
    v = M.compute(_manifest(_cells(
        [4.80, 4.90], [4.82, 4.92], [4.86, 4.94], [4.80, 4.90],
    )))
    assert v["interaction"] == pytest.approx(v["mean_matched"] - v["mean_mismatched"])


def test_a_missing_cell_is_an_error_not_a_zero():
    cells = _cells([4.8, 4.8], [4.8, 4.8], [4.8, 4.8], [4.8, 4.8])
    del cells["treatment-s1339|eval_masked"]
    with pytest.raises(KeyError, match="missing from the manifest"):
        M.compute(_manifest(cells))


def test_a_missing_file_exits_two(tmp_path, capsys):
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "tools/x17_mask2x2.py", "--file", str(tmp_path / "nope.json")],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert r.returncode == 2
    assert "has the 2x2 kernel been archived" in r.stdout
