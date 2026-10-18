"""X17's design, pinned where it is cheap to check.

The kernel asserts its own preconditions on Kaggle, which is the right place for
the expensive ones (does the mask engage, do the configs differ by one key). But
an assertion that only fires after GPU time is a bad place to discover a design
drift, so the parts that are checkable from the repo are checked here.

The registration this implements: `2026-09-28-X17-doc-mask.md`, amended by
`2026-10-18-X17-amendment-1.md`. Seed 1337 is excluded (it generated the
hypotheses); the DM-1 bar is the original's, unchanged.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent
KERNEL = ROOT / "tools" / "kalia_x17_kernel.py"


def _constants() -> dict:
    """Execute the kernel's module-level constants without running the kernel.

    `RUNS` is a comprehension, so this cannot be `ast.literal_eval`; the
    assignment nodes before the first `def` are compiled and run instead. Nothing
    after that point is touched, so no Kaggle path is resolved and no training
    happens.
    """
    body = []
    for node in ast.parse(KERNEL.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            break
        if isinstance(node, ast.Assign):
            body.append(node)
    ns: dict = {"os": __import__("os")}
    exec(compile(ast.Module(body=body, type_ignores=[]), "<kernel>", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def const() -> dict:
    return _constants()


def test_the_two_configs_differ_in_exactly_one_key():
    """The entire design constraint: one variable. If a hyperparameter moved the
    comparison would confound masking with whatever else changed, which is how
    X18's arm drifted from its control."""
    base = yaml.safe_load((ROOT / "configs" / "micro-base.yaml").read_text())
    mask = yaml.safe_load((ROOT / "configs" / "micro-docmask.yaml").read_text())
    assert base["model"] == mask["model"]
    changed = {
        k for k in set(base["train"]) | set(mask["train"])
        if base["train"].get(k) != mask["train"].get(k)
    }
    assert changed == {"doc_mask"}
    assert base["train"]["doc_mask"] is False
    assert mask["train"]["doc_mask"] is True


def test_the_run_order_alternates_within_each_seed(const):
    """Control then treatment inside each seed, both seeds in one kernel, so a
    machine that drifts loses both arms of a pair equally."""
    order = [(run, cfg, seed) for run, cfg, seed in const["RUNS"]]
    assert [seed for _, _, seed in order] == ["1338", "1338", "1339", "1339"]
    assert [run.split("-s")[0] for run, _, _ in order] == [
        "control", "treatment", "control", "treatment",
    ]
    assert [cfg for _, cfg, _ in order] == [
        "micro-base", "micro-docmask", "micro-base", "micro-docmask",
    ]
    # every run name is unique -- ablate.py writes out/<arm>, so a collision
    # would silently overwrite one arm with another
    assert len({run for run, _, _ in order}) == len(order)


def test_seed_1337_is_excluded(const):
    """It generated every prior hypothesis in this project; X21's registration
    excluded it as circular and D48 flagged it as unusually good."""
    assert "1337" not in const["SEEDS"]
    assert const["SEEDS"] == ("1338", "1339")


def test_the_dm1_bar_is_the_originals_number(const):
    """The amendment deliberately did not move it: "we are not moving the
    goalposts for a fix we already believe in"."""
    assert const["DM1_BAR"] == -0.010


def test_the_kernel_only_pins_what_it_declares(tmp_path):
    """X17 runs four arms from one kernel, so it has no per-arm SEED to stage.
    The stager must pin exactly ARM, KERNEL_ID and KERNEL_REVISION -- inventing
    a SEED pin would imply a knob the run does not have."""
    r = subprocess.run(
        [sys.executable, "tools/stage_x29_arm.py", "docmask", "--kernel", "x17",
         "--dest", str(tmp_path), "--allow-dirty"],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    staged = (tmp_path / "kalia_x17_kernel.py").read_text()
    for name in ("ARM", "KERNEL_ID", "KERNEL_REVISION"):
        assert "unknown" not in staged.split(f'{name} = "')[1].split('"')[0]
    # no SEED constant exists in this runner, and none was invented
    assert 'SEED = "' not in staged
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == "subhajitlucky/kalia-x17-docmask"
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is True
