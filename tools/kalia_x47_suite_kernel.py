"""Kaggle CPU kernel: v0.3.1 on the standard 50-500M zero-shot suite.

Tasks: ARC-Easy, ARC-Challenge, BoolQ, HellaSwag, LAMBADA, OpenBookQA,
PIQA, SciQ, WinoGrande. 2,000 samples where available. CPU only; no GPU.

This is the Stage 0 baseline for the next base-model data experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_REPO = "kalia-lm/kalia-v031"
BASE_SHA = "bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468"
TASKS = "arc_easy,arc_challenge,boolq,hellaswag,lambada_openai,openbookqa,piqa,sciq,winogrande"
LIMIT = 2000

hits = sorted(Path("/kaggle/input").rglob("eval_bench.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached or lacks eval_bench.py: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))
for req in ("eval_bench.py", "kalia_lm.py", "eval_reversibility.py",
            "model.py", "data.py"):
    assert (work / req).exists(), f"{req} missing"
print("code:", code_root)

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
assert base_sha == BASE_SHA, (base_sha, BASE_SHA)
raw_cfg = json.loads(Path(cf).read_text())
cfg = {k: v for k, v in raw_cfg.items() if k in GPTConfig.__dataclass_fields__}
model = GPT(GPTConfig(**cfg))
load = model.load_state_dict(load_file(sf), strict=False)
assert not load.missing_keys and not load.unexpected_keys
ckpt = Path("/kaggle/working/base_v031.pt")
torch.save({"model": model.state_dict(), "config": {"model": cfg},
            "step": 0, "tokens": 0}, ckpt)
print("base sha256:", base_sha)

try:
    import lm_eval  # noqa: F401
except Exception:
    print("installing lm-eval", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lm-eval"],
                   check=True)

out = Path("/kaggle/working/x47_suite.json")
subprocess.run(
    [sys.executable, "eval_bench.py", "--ckpt", str(ckpt),
     "--limit", str(LIMIT), "--tasks", TASKS, "--out", str(out)],
    check=True,
)
results = json.loads(out.read_text())
summary = {
    "experiment": "X47-stage0-suite",
    "base_repo": BASE_REPO,
    "base_sha256": base_sha,
    "limit": LIMIT,
    "tasks": TASKS,
    "results": results,
}
Path("/kaggle/working/x47_stage0_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)

print("\nX47 STAGE 0 — v0.3.1 STANDARD SUITE")
for task, row in sorted(results.items()):
    acc = row.get("acc,none")
    se = row.get("acc_stderr,none")
    n = row.get("n-samples", {}).get("original") or row.get("n-samples", {}).get("effective")
    print(f"  {task:18s} acc {acc:.4f} se {se:.4f} n {n}")
print("  summary: /kaggle/working/x47_stage0_summary.json")
