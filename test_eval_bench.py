"""The benchmark sample size, which stayed at 500 through two releases.

D45 measured lm-eval's standard error at `limit=500` at 1.99–2.24 pp — wider than
the movements the v0.2.0 evaluation reported and wider than the 1.0 pp stability
allowance it was judged against. The 2026-10-17 review recommended raising it and
nothing changed, because the number was set in the notebook builders, not in the
tool, so changing the tool's default would have done nothing at all.

These tests pin the policy, not the harness. `eval_bench` imports `lm_eval` and
`kalia_lm` inside `main` precisely so this file can run without either.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load(name: str):
    path = ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load("eval_bench")


def test_the_default_is_above_the_measured_floor():
    """2,000 samples halves the standard error against 500 at zero GPU cost."""
    args = bench.build_parser().parse_args([])
    assert args.limit == 2000
    assert args.limit >= bench.MIN_REPORTABLE_LIMIT
    assert bench.limit_warning(args.limit) is None


def test_the_historical_size_warns_rather_than_passing_silently():
    """500 is what every recorded number used. Reproducing one is legitimate;
    doing it without noticing is the defect."""
    warning = bench.limit_warning(500)
    assert warning is not None
    assert "500" in warning and "D45" in warning


def test_the_floor_itself_does_not_warn():
    assert bench.limit_warning(bench.MIN_REPORTABLE_LIMIT) is None
    assert bench.limit_warning(bench.MIN_REPORTABLE_LIMIT - 1) is not None


def test_the_module_imports_without_the_harness_installed():
    """The reason the imports moved: `lm_eval` is not in requirements.txt, so a
    module-level import made this file untestable in CI."""
    assert "lm_eval" not in (ROOT / "requirements.txt").read_text()
    assert bench.MIN_REPORTABLE_LIMIT == 1000
