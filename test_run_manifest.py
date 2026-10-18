"""One run record, two kernels: the provenance block must not drift.

A Kaggle kernel script cannot import from this repo -- it runs against the
``kalia-code-dev`` snapshot, which lags HEAD -- so the manifest helper is
duplicated in the two runner scripts rather than shared. This test is what stops
the copies diverging: the same shape as the cohort/kernel_sources contract, two
files holding one fact, so the fact gets pinned.

The record exists because of the 2026-10-17 review's finding (4): the seed-1402
replicas could not state their own seed or the code revision they ran, and both
had to be re-derived from the staging scripts afterwards.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
RUNNERS = ("kalia_x29_kernel.py", "kalia_kautilya_trio_kernel.py")
HELPERS = ("sha256_file", "code_fingerprint", "write_run_manifest")


def _load_stager():
    path = ROOT / "tools" / "stage_x29_arm.py"
    spec = importlib.util.spec_from_file_location("stage_x29_arm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stager = _load_stager()


def _helpers(carrier: str) -> dict:
    tree = ast.parse((ROOT / "tools" / carrier).read_text())
    return {
        node.name: ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS
    }


@pytest.mark.parametrize("carrier", RUNNERS)
def test_runner_defines_the_whole_provenance_block(carrier):
    have = _helpers(carrier)
    assert set(HELPERS) <= set(have), f"{carrier} is missing {sorted(set(HELPERS) - set(have))}"


def test_the_two_copies_are_identical():
    first, second = (_helpers(c) for c in RUNNERS)
    for name in HELPERS:
        assert first[name] == second[name], (
            f"{name} has drifted between {RUNNERS[0]} and {RUNNERS[1]}; "
            "fix both, they are the same record"
        )


@pytest.mark.parametrize("carrier", RUNNERS)
def test_every_runner_writes_a_manifest_naming_its_seed(carrier):
    """A manifest that does not name the seed would not have caught the thing
    that motivated it."""
    src = (ROOT / "tools" / carrier).read_text()
    assert "write_run_manifest(" in src
    assert "seed=int(SEED)" in src, f"{carrier} writes a manifest without the seed"


@pytest.mark.parametrize("carrier", RUNNERS)
def test_manifest_helpers_read_the_pinned_constants(carrier):
    """kernel_id and kernel_revision are pinned by the stager; a helper that
    hardcodes them would record a provenance nobody checked."""
    src = (ROOT / "tools" / carrier).read_text()
    assert 'KERNEL_ID = "unknown"' in src
    assert 'KERNEL_REVISION = "unknown"' in src
    block = _helpers(carrier)["write_run_manifest"]
    assert "KERNEL_ID" in block and "KERNEL_REVISION" in block


@pytest.mark.parametrize("carrier", RUNNERS)
def test_the_no_replay_control_cannot_pick_up_a_replay_shard(carrier):
    """The control's whole claim is that replay is the only variable that
    moved. The X29 runner must not be able to hand train.py a replay shard at
    replay_prob 0 -- or to reach the carve at all."""
    if carrier != "kalia_x29_kernel.py":
        pytest.skip("the Kautilya runner has no replay knob to disable")
    src = (ROOT / "tools" / carrier).read_text()
    assert "if replay_prob > 0:" in src, "the replay carve is not gated on the fraction"
    assert "if replay_bin:" in src, "cmd would carry --replay-bin unconditionally"
    assert 'cmd += ["--replay-bin", replay_bin]' in src


# --- the arm/replay pairing, checked against the stager's own table ----------
# The stager decides which fraction each arm stages with; the kernel asserts the
# pairing is coherent. If those two ever disagree, one of them is wrong and the
# run is mislabeled -- so the kernel's guard is executed here on the stager's
# table rather than trusted.

X29 = "kalia_x29_kernel.py"


def _constant_block(source: str):
    """The module-level constants and guards, without the Kaggle-side effects."""
    body = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            break
        if isinstance(node, (ast.Assign, ast.Assert)):
            body.append(node)
    assert body, "no module-level constants found; did the runner's shape change?"
    return ast.Module(body=body, type_ignores=[])


def _derive_kernel_constants(arm: str, replay: str) -> dict:
    src = (ROOT / "tools" / X29).read_text()
    src = re.sub(r'^ARM = ".*?"', f'ARM = "{arm}"', src, count=1, flags=re.M)
    src = re.sub(r'^REPLAY = ".*?"', f'REPLAY = "{replay}"', src, count=1, flags=re.M)
    ns: dict = {}
    exec(compile(_constant_block(src), "<kernel>", "exec"), ns)
    return ns


@pytest.mark.parametrize("arm,replay", sorted(_stager.KERNELS["x29"]["arms"].items()))
def test_kernel_guard_accepts_every_pairing_the_stager_makes(arm, replay):
    ns = _derive_kernel_constants(arm, replay)
    assert ns["replay_prob"] == float(replay)
    ns["replay_pattern"] in ("uniform", "krama", "jata", "ghana")


@pytest.mark.parametrize("arm,replay", [("noreplay", "0.10"), ("uniform", "0.00")])
def test_kernel_guard_rejects_a_contradictory_pairing(arm, replay):
    """A control named `noreplay` that carries replay, or a ladder arm with
    none, is a mislabeled run; the kernel refuses it as well as the stager."""
    with pytest.raises(AssertionError):
        _derive_kernel_constants(arm, replay)
