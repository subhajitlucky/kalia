"""Kaggle CPU kernel: X41 baseline on the clean 2x2 composition benchmark.

Scores the public v0.3.1 model (58M) in the normal instruction+prefix mode and
in the gold-edit mode, on the full 1,600-record benchmark (400 per cell). No
training, no GPU. This is the baseline for the X41 edit-compiler experiment.

GitHub artifacts only; the public model is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BENCH_SEED = 2026
BENCH_SHA = "aad1b32187a8697b896f6be8408b979abd389466bd9fd42256b2f547e87dd45b"
BASE_REPO = "kalia-lm/kalia-v031"

# --- code -------------------------------------------------------------------
hits = sorted(Path("/kaggle/input").rglob("eval_composition_benchmark.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached or lacks eval_composition_benchmark.py: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))
for req in ("composition_benchmark.py", "eval_composition_benchmark.py",
            "intervention_pairs.py", "eval_ip.py", "eval_reversibility.py",
            "model.py"):
    assert (work / req).exists(), f"{req} missing from code dataset"
print("code:", code_root)
print("work:", work)

# --- base --------------------------------------------------------------------
try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "safetensors", "huggingface_hub"], check=True)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

import torch  # noqa: E402

from model import GPT, GPTConfig  # noqa: E402

torch.set_num_threads(min(8, os.cpu_count() or 1))
base_dir = Path("/kaggle/working/base")
base_dir.mkdir(parents=True, exist_ok=True)
sf = hf_hub_download(BASE_REPO, "model.safetensors", local_dir=str(base_dir))
cf = hf_hub_download(BASE_REPO, "config.json", local_dir=str(base_dir))
base_sha = hashlib.sha256(Path(sf).read_bytes()).hexdigest()
raw_cfg = json.loads(Path(cf).read_text())
cfg = {k: v for k, v in raw_cfg.items() if k in GPTConfig.__dataclass_fields__}
model = GPT(GPTConfig(**cfg))
load = model.load_state_dict(load_file(sf), strict=False)
assert not load.missing_keys and not load.unexpected_keys, (
    load.missing_keys, load.unexpected_keys
)
ckpt = Path("/kaggle/working/base_v031.pt")
torch.save({"model": model.state_dict(), "config": {"model": cfg},
            "step": 0, "tokens": 0}, ckpt)
print("base:", sf)
print("base sha256:", base_sha)
print("params:", sum(p.numel() for p in model.parameters()))

# --- benchmark ---------------------------------------------------------------
bench_dir = Path("/kaggle/working/x41-bench")
if not bench_dir.exists():
    subprocess.run([sys.executable, "composition_benchmark.py",
                    "--out-dir", str(bench_dir), "--seed", str(BENCH_SEED)],
                   check=True)
bench = bench_dir / "bench.jsonl"
bench_sha = hashlib.sha256(bench.read_bytes()).hexdigest()
print("bench sha256:", bench_sha)
assert bench_sha == BENCH_SHA, (bench_sha, BENCH_SHA)
audit = json.loads((bench_dir / "audit.json").read_text())
print("bench cells:", {k: v["n"] for k, v in audit["cells"].items()})

# --- eval --------------------------------------------------------------------
summary = {
    "experiment": "X41-baseline",
    "base_repo": BASE_REPO,
    "base_sha256": base_sha,
    "base_params": sum(p.numel() for p in model.parameters()),
    "bench_sha256": bench_sha,
    "bench_seed": BENCH_SEED,
    "runs": {},
}
for mode, tag in (("instruction_prefix", "normal"), ("edited_prefix", "gold")):
    out = Path(f"/kaggle/working/x41_base_{tag}.json")
    rows = Path(f"/kaggle/working/x41_base_{tag}_rows.jsonl")
    subprocess.run([sys.executable, "eval_composition_benchmark.py",
                    "--ckpt", str(ckpt), "--bench", str(bench),
                    "--out", str(out), "--rows", str(rows),
                    "--device", "cpu", "--batch-size", "16",
                    "--prompt-mode", mode], check=True)
    payload = json.loads(out.read_text())
    summary["runs"][tag] = {
        "prompt_mode": mode,
        "by_cell": payload["by_cell"],
        "by_operation": payload["by_operation"],
        "overall": payload["overall"],
        "n": payload["n"],
    }

Path("/kaggle/working/x41_baseline_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)

print("\nX41 BASELINE")
for tag in ("normal", "gold"):
    print(f"  {tag}:")
    for cell, row in summary["runs"][tag]["by_cell"].items():
        print(f"    {cell}: {row['accuracy']:.4f} (chance {row['chance']:.2f}, "
              f"n={row['n']})")
    for op, row in sorted(summary["runs"][tag]["by_operation"].items()):
        print(f"    {op:16s}: {row['accuracy']:.4f} (n={row['n']})")
print("  summary: /kaggle/working/x41_baseline_summary.json")
