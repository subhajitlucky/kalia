"""The CL-0 probe, which had never been executed before this.

It is registered as Step 0 of v0.3.0 with a hard prerequisite flag, and it had
never been run. Running it on a tiny CPU checkpoint found a bug that unit tests
could not have: the bpB field divided by a hardcoded 4.4086 bytes-per-token,
while our corpus measures ~3.38. Every bpB the forgetting ledger had ever
reported was ~30% too low, silently, and that ledger is what Steps 2-4 get scored
against.

These tests run the real script end to end.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent


def _corpus(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    (data / "probe").mkdir(parents=True)
    for name, n, seed in (
        ("train.bin", 20_000, 1),
        ("val.bin", 8_000, 2),
        ("probe/val_forget.bin", 8_000, 3),
    ):
        p = data / name
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.randint(0, 256, (n,), generator=torch.Generator().manual_seed(seed)).to(
            torch.uint16).numpy().tofile(p)
    return data


def _ckpt(tmp_path: Path, data: Path) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    r = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--config", str(cfg_path),
         "--data-dir", str(data), "--out-dir", str(tmp_path / "out"),
         "--max-steps", "4", "--seed", "1"],
        capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return tmp_path / "out" / "ckpt.pt"


def _probe(ckpt, data, ledger, extra=()):
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / "forgetting_probe.py"),
         "--ckpt", str(ckpt), "--probe", str(data / "probe" / "val_forget.bin"),
         "--ledger", str(ledger), "--batches", "2", "--batch-size", "4",
         "--device", "cpu", *extra],
        capture_output=True, text=True, cwd=ROOT, timeout=600)


def test_probe_runs_and_writes_a_ledger_row(tmp_path):
    """It had never been executed. This is the first run in the project's history."""
    data = _corpus(tmp_path)
    ckpt = _ckpt(tmp_path, data)
    ledger = tmp_path / "forgetting.jsonl"
    r = _probe(ckpt, data, ledger)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    rec = rows[0]
    for key in ("step", "probe_loss", "probe_tokens", "seed", "recorded"):
        assert key in rec, key
    assert rec["step"] == 4
    assert rec["probe_loss"] > 0


def test_bpb_is_measured_not_hardcoded(tmp_path):
    """The bug: a hardcoded divisor made every bpB wrong by ~30%."""
    data = _corpus(tmp_path)
    ckpt = _ckpt(tmp_path, data)
    ledger = tmp_path / "f.jsonl"
    assert _probe(ckpt, data, ledger).returncode == 0
    rec = json.loads(ledger.read_text().splitlines()[0])
    assert "bytes_per_token" in rec, "the divisor must be recorded with the value"
    bpt = rec["bytes_per_token"]
    assert bpt and bpt > 1.0, bpt
    # And bpB must actually be consistent with the divisor it reports.
    import math
    assert rec["bpB"] == pytest.approx(rec["probe_loss"] / math.log(2) / bpt, rel=1e-4)
    # The old constant would have given a visibly different number.
    assert rec["bpB"] != pytest.approx(rec["probe_loss"] / math.log(2) / 4.4086, rel=1e-3)


def test_no_bpb_flag_omits_bpb_rather_than_inventing_it(tmp_path):
    data = _corpus(tmp_path)
    ckpt = _ckpt(tmp_path, data)
    ledger = tmp_path / "f.jsonl"
    r = _probe(ckpt, data, ledger, extra=("--no-bpb",))
    assert r.returncode == 0
    rec = json.loads(ledger.read_text().splitlines()[0])
    assert rec["bpB"] is None
    assert rec["bytes_per_token"] is None


def test_drift_is_relative_to_the_best_ever_not_the_previous(tmp_path):
    """Two points; drift must be measured against the minimum, so it is <= 0."""
    data = _corpus(tmp_path)
    ckpt = _ckpt(tmp_path, data)
    ledger = tmp_path / "f.jsonl"
    assert _probe(ckpt, data, ledger).returncode == 0
    assert _probe(ckpt, data, ledger, extra=("--step", "8")).returncode == 0
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    assert rows[0]["drift_from_best"] is None, "the first point has nothing to compare to"
    assert rows[1]["drift_from_best"] is not None
    best = min(r["probe_loss"] for r in rows)
    assert rows[1]["drift_from_best"] == pytest.approx(rows[1]["probe_loss"] - best, abs=1e-4)


def test_report_only_mode_needs_no_checkpoint(tmp_path):
    data = _corpus(tmp_path)
    ckpt = _ckpt(tmp_path, data)
    ledger = tmp_path / "f.jsonl"
    assert _probe(ckpt, data, ledger).returncode == 0
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "forgetting_probe.py"), "--ledger", str(ledger)],
        capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "best ever" in r.stdout


def test_report_only_on_an_empty_ledger_says_so(tmp_path):
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "forgetting_probe.py"),
         "--ledger", str(tmp_path / "empty.jsonl")],
        capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert r.returncode == 0
    assert "empty" in r.stdout.lower()
