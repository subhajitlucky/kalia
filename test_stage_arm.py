"""The stager exists so a launch cannot silently run the wrong arm or seed.

`stage_x29_arm.py` pins ``ARM``, ``SEED``, ``REPLAY`` (where the runner has
one), ``KERNEL_ID`` and ``KERNEL_REVISION`` in a copy of the runner and verifies
the committed metadata agrees. Forgetting any one of them burns a GPU run -- or
worse, produces a result labelled as an arm it is not. The patch semantics are
pinned here, on the real runner files, and every registered arm is staged.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent


def _load(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stager = _load("stage_x29_arm")
KERNELS = stager.KERNELS


def _stage(dest: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        # --allow-dirty: the dirty guard is about a real push, not about these
        # tests, and a working tree mid-edit must not turn the suite red.
        [sys.executable, "tools/stage_x29_arm.py", *args, "--dest", str(dest), "--allow-dirty"],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )


def _pinned(src: str, name: str) -> str:
    m = re.search(rf'^{name} = "(.*?)"', src, re.M)
    assert m, f"{name} is not pinned in the staged runner"
    return m.group(1)


def test_uniform_replica_pins_arm_seed_and_replay(tmp_path):
    r = _stage(tmp_path, "uniform", "--seed", "1402")
    assert r.returncode == 0, r.stderr
    src = (tmp_path / "kalia_x29_kernel.py").read_text()
    assert _pinned(src, "ARM") == "uniform"
    assert _pinned(src, "SEED") == "1402"
    assert _pinned(src, "REPLAY") == "0.10"
    assert '"seed": int(SEED)' in src, "the config must use the patched seed"
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == "subhajitlucky/kalia-x29-uniform-s1402"
    assert meta["code_file"] == "kalia_x29_kernel.py"
    assert meta["is_private"] is True


@pytest.mark.parametrize("seed,suffix", [("1401", ""), ("1402", "-s1402")])
def test_noreplay_control_pins_replay_to_zero(tmp_path, seed, suffix):
    """The 2026-10-18 control: same runner, corpus and budget as `uniform`,
    replay at zero. If this ever stages at 0.10 it is not a control."""
    r = _stage(tmp_path, "noreplay", "--seed", seed)
    assert r.returncode == 0, r.stderr
    src = (tmp_path / "kalia_x29_kernel.py").read_text()
    assert _pinned(src, "ARM") == "noreplay"
    assert _pinned(src, "REPLAY") == "0.00"
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == f"subhajitlucky/kalia-x29-noreplay{suffix}"


def test_the_patha_ladder_keeps_its_registered_replay_fraction(tmp_path):
    for arm in ("uniform", "krama", "jata", "ghana"):
        assert KERNELS["x29"]["arms"][arm] == "0.10", arm
    r = _stage(tmp_path, "krama")
    assert r.returncode == 0, r.stderr
    assert _pinned((tmp_path / "kalia_x29_kernel.py").read_text(), "REPLAY") == "0.10"


def test_static_replica_uses_the_trio_runner(tmp_path):
    r = _stage(tmp_path, "static", "--kernel", "x28", "--seed", "1402")
    assert r.returncode == 0, r.stderr
    src = (tmp_path / "kalia_kautilya_trio_kernel.py").read_text()
    assert _pinned(src, "ARM") == "static"
    assert _pinned(src, "SEED") == "1402"
    # The trio samples per-source shards and passes no --replay-bin; there is
    # no REPLAY constant to pin, and inventing one would imply a knob it lacks.
    assert not re.search(r'^REPLAY = "', src, re.M)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == "subhajitlucky/kalia-x28-static-s1402"


def test_original_seed_keeps_the_original_kernel_id(tmp_path):
    r = _stage(tmp_path, "krama")
    assert r.returncode == 0, r.stderr
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == "subhajitlucky/kalia-x29-krama"
    assert _pinned((tmp_path / "kalia_x29_kernel.py").read_text(), "SEED") == "1401"


def test_kernel_revision_and_id_are_stamped_from_real_sources(tmp_path):
    """The manifest's provenance has to describe the code that runs. Before
    2026-10-18 the runners wrote no manifest at all: the seed-1402 replicas
    could not say which seed they ran, and it had to be re-derived by staging
    the arm again."""
    r = _stage(tmp_path, "noreplay")
    assert r.returncode == 0, r.stderr
    src = (tmp_path / "kalia_x29_kernel.py").read_text()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=ROOT, check=True).stdout.strip()
    revision = _pinned(src, "KERNEL_REVISION")
    assert revision.startswith(head), f"{revision} does not name HEAD"
    assert set(revision) <= set("0123456789abcdef+dirty")
    assert _pinned(src, "KERNEL_ID") == "subhajitlucky/kalia-x29-noreplay"


def test_arm_not_in_the_selected_kernel_is_rejected(tmp_path):
    r = _stage(tmp_path, "static")  # static belongs to x28, not x29
    assert r.returncode != 0
    assert "arms are" in r.stderr


def test_every_registered_arm_has_committed_metadata():
    for kernel, spec in KERNELS.items():
        for arm in spec["arms"]:
            path = ROOT / "kaggle" / f"kernel-metadata-{kernel}-{arm}.json"
            assert path.exists(), f"{kernel} {arm} is stageable in the script but has no {path.name}"


def _stageable_cases() -> list[tuple[str, str, str]]:
    """Every committed metadata file this stager owns, as (kernel, arm, seed).

    Derived from the files rather than from the arm lists, so a replica whose
    metadata was committed but whose arm was never made stageable -- or the
    reverse -- shows up as a failure instead of as a missing test.
    """
    cases = set()
    for path in sorted((ROOT / "kaggle").glob("kernel-metadata-*.json")):
        stem = path.stem[len("kernel-metadata-"):]
        for kernel, spec in KERNELS.items():
            if not stem.startswith(f"{kernel}-"):
                continue
            arms = "|".join(sorted(map(re.escape, spec["arms"]), key=len, reverse=True))
            m = re.fullmatch(rf"({arms})(?:-s(\d+))?", stem[len(kernel) + 1:])
            if m:
                cases.add((kernel, m.group(1), m.group(2) or "1401"))
    return sorted(cases)


@pytest.mark.parametrize("kernel,arm,seed", _stageable_cases())
def test_every_committed_arm_metadata_stages(tmp_path, kernel, arm, seed):
    """A renamed constant, or an arm whose metadata was never committed, must
    fail here rather than after a GPU run has been spent."""
    r = _stage(tmp_path, arm, "--kernel", kernel, "--seed", seed)
    assert r.returncode == 0, f"{kernel} {arm} seed {seed}: {r.stderr}"
