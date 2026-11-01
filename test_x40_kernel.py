"""X40: the 30M kernel and its staging contract.

The kernel is the expensive path, so the cheap checks happen first: the source
must compile, the arm/seed grid and bars must match the registration, the
metadata must agree with the runner, and the stager must pin the identity it
claims to pin.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KERNEL = ROOT / "tools" / "kalia_x40_ip_kernel.py"
META = ROOT / "kaggle" / "kernel-metadata-x40-ip.json"
STAGE = ROOT / "tools" / "stage_x40_ip.py"


def test_kernel_compiles():
    compile(KERNEL.read_text(), str(KERNEL), "exec")


def test_kernel_grid_matches_the_registration():
    src = KERNEL.read_text()
    assert 'SEEDS = ("1338", "1339")' in src
    assert 'ARMS = ("control", "adapter", "pairs", "workspace", "ip")' in src
    assert 'PAIR_ARMS = ("pairs", "workspace", "ip")' in src
    assert "STEPS = 763" in src
    assert "EVENT_SLOTS = 16" in src
    assert "EVENT_SLOT_DIM = 64" in src
    assert "IP_STATE_WEIGHT = 0.1" in src
    assert "IP_DIV_WEIGHT = 0.01" in src
    assert "EXPECTED_CONTROL = 29_921_280" in src
    assert "I3_MARGIN = 0.10" in src
    assert "I5_ENTROPY = 2.0" in src
    assert "I5_RANK = 4.0" in src
    assert "I5_PROBE = 0.10" in src


def test_kernel_uses_the_fixed_scorer_and_writes_the_verdict():
    src = KERNEL.read_text()
    assert "summarise_arm" in src
    assert "x40_seen_unseen_rows.jsonl" in src
    assert "x40_summary.json" in src
    assert "unseen_delta_ip_minus_pairs" in src
    assert "workspace_stats" not in src or "summarise_arm" in src


def test_metadata_matches_the_runner():
    meta = json.loads(META.read_text())
    assert meta["id"] == "subhajitlucky/kalia-x40-ip"
    assert meta["code_file"] == KERNEL.name
    assert meta["enable_gpu"] is True
    assert "subhajitlucky/kalia-code-dev" in meta["dataset_sources"]
    assert "subhajitlucky/kalia-prep-v2b" in meta["kernel_sources"]


def test_stager_pins_identity(tmp_path):
    result = subprocess.run(
        [sys.executable, str(STAGE), "--dest", str(tmp_path), "--allow-dirty"],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    staged = (tmp_path / KERNEL.name).read_text()
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert f'KERNEL_ID = "{meta["id"]}"' in staged
    assert 'KERNEL_ID = "unknown"' not in staged
    assert 'KERNEL_REVISION = "unknown"' not in staged
    assert "KERNEL_REVISION =" in staged
