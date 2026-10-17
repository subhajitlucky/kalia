"""The stager exists so a launch cannot silently run the wrong arm or seed.

`stage_x29_arm.py` patches ``ARM`` and ``SEED`` in a copy of the runner and
verifies the committed metadata agrees. Forgetting either patch burns a GPU
run (or worse, produces a result labelled as an arm it is not) — so the patch
semantics are pinned here, on the real runner files.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _stage(dest: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "tools/stage_x29_arm.py", *args, "--dest", str(dest)],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )


def test_uniform_replica_patches_arm_and_seed(tmp_path):
    r = _stage(tmp_path, "uniform", "--seed", "1402")
    assert r.returncode == 0, r.stderr
    src = (tmp_path / "kalia_x29_kernel.py").read_text()
    assert 'ARM = "uniform"' in src
    assert 'SEED = "1402"' in src
    assert '"seed": int(SEED)' in src, "the config must use the patched seed"
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == "subhajitlucky/kalia-x29-uniform-s1402"
    assert meta["code_file"] == "kalia_x29_kernel.py"
    assert meta["is_private"] is True


def test_static_replica_uses_the_trio_runner(tmp_path):
    r = _stage(tmp_path, "static", "--kernel", "x28", "--seed", "1402")
    assert r.returncode == 0, r.stderr
    src = (tmp_path / "kalia_kautilya_trio_kernel.py").read_text()
    assert 'ARM = "static"' in src
    assert 'SEED = "1402"' in src
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == "subhajitlucky/kalia-x28-static-s1402"


def test_original_seed_keeps_the_original_kernel_id(tmp_path):
    r = _stage(tmp_path, "krama")
    assert r.returncode == 0, r.stderr
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == "subhajitlucky/kalia-x29-krama"
    assert 'SEED = "1401"' in (tmp_path / "kalia_x29_kernel.py").read_text()


def test_arm_not_in_the_selected_kernel_is_rejected(tmp_path):
    r = _stage(tmp_path, "static")  # static belongs to x28, not x29
    assert r.returncode != 0
    assert "arms are" in r.stderr
