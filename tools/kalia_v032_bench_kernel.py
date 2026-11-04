"""Kaggle CPU kernel: V032 2k benchmark guard.

Runs the canonical 2,000-sample benchmark pair on the base v0.3.1 and the V032
adapter checkpoints in the same kernel, so the comparison is paired. V-5:
no task may regress beyond 2σ versus the base.

GitHub only; no Hugging Face push.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

KERNEL_ID = "unknown"
KERNEL_REVISION = "unknown"
BASE_REPO = "kalia-lm/kalia-v031"
LIMIT = 2000
TASKS = "arc_easy,piqa,hellaswag,lambada_openai,winogrande"

# Published v0.3.1 2k reference, for a cross-check only. The paired base run
# below is the comparison the bar uses.
BASE_2K = {
    "arc_easy": {"acc": 0.4800, "se": 0.011174185930778483},
    "hellaswag": {"acc": 0.3235, "se": 0.010463202870400323},
    "lambada_openai": {"acc": 0.2700, "se": 0.009929718004547002,
                       "ppl": 122.88354952002013, "ppl_se": 10.672943965144498},
    "piqa": {"acc": 0.6196953210010882, "se": 0.011326620892570301},
    "winogrande": {"acc": 0.5224940805051302, "se": 0.014038257824059859},
}

# --- code -------------------------------------------------------------------
hits = sorted(Path("/kaggle/input").rglob("eval_bench.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, "kalia-code-dev not attached or lacks eval_bench.py"
code_root = str(preferred[0].parent)
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("eval_bench.py", "kalia_lm.py", "eval_reversibility.py",
            "model.py", "data.py"):
    assert os.path.exists(req), f"{req} missing"
print("code:", code_root)

# --- base and adapters -------------------------------------------------------
try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "safetensors", "huggingface_hub"], check=True)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

from model import GPT, GPTConfig  # noqa: E402

base_dir = Path("/kaggle/working/base")
base_dir.mkdir(parents=True, exist_ok=True)
sf = hf_hub_download(BASE_REPO, "model.safetensors", local_dir=str(base_dir))
cf = hf_hub_download(BASE_REPO, "config.json", local_dir=str(base_dir))
base_sd = load_file(sf)
base_sha = hashlib.sha256(Path(sf).read_bytes()).hexdigest()
raw_cfg = json.loads(Path(cf).read_text())
cfg = {k: v for k, v in raw_cfg.items() if k in GPTConfig.__dataclass_fields__}
print("base:", sf, "sha", base_sha[:16])

adapter_hits = sorted(Path("/kaggle/input").rglob("adapter_seed*.pt"))
adapter_hits = [p for p in adapter_hits if "kalia-v032-adapter" in str(p)] or adapter_hits
assert adapter_hits, "attach subhajitlucky/kalia-v032-adapter"
print("adapters:", [str(p) for p in adapter_hits])


def build_ckpt(adapter_path: Path | None, out: Path) -> dict:
    if adapter_path is None:
        model_cfg = dict(cfg)
        model = GPT(GPTConfig(**model_cfg))
        load = model.load_state_dict(base_sd, strict=False)
        assert not load.missing_keys and not load.unexpected_keys
    else:
        model_cfg = dict(cfg)
        model_cfg["memory_control"] = True
        model_cfg["memory_control_hidden"] = 1024
        model = GPT(GPTConfig(**model_cfg))
        load = model.load_state_dict(base_sd, strict=False)
        assert all(k.startswith("adapter.") for k in load.missing_keys), load.missing_keys
        assert not load.unexpected_keys
        payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
        load = model.load_state_dict(payload["adapter"], strict=False)
        assert not load.missing_keys and not load.unexpected_keys, (
            load.missing_keys, load.unexpected_keys
        )
    torch.save({
        "model": model.state_dict(),
        "config": {"model": model_cfg},
        "step": 2000,
        "tokens": 0,
    }, out)
    return {"ckpt": str(out), "bytes": out.stat().st_size,
            "adapter": str(adapter_path) if adapter_path else None}


def run_bench(ckpt: Path, out: Path) -> dict:
    subprocess.run(
        [sys.executable, "eval_bench.py", "--ckpt", str(ckpt),
         "--limit", str(LIMIT), "--tasks", TASKS, "--out", str(out)],
        check=True,
    )
    return json.loads(out.read_text())


def metrics(results: dict) -> dict:
    out = {}
    for task, row in results.items():
        acc = row.get("acc,none")
        se = row.get("acc_stderr,none", 0.0)
        out[task] = {"acc": acc, "se": se}
        if "perplexity,none" in row:
            out[task]["ppl"] = row["perplexity,none"]
            out[task]["ppl_se"] = row.get("perplexity_stderr,none", 0.0)
    return out


def compare(base: dict, adapter: dict) -> dict:
    comparisons = {}
    for task in base:
        b, a = base[task], adapter.get(task, {})
        delta = a.get("acc", 0.0) - b.get("acc", 0.0)
        sigma = (b.get("se", 0.0) ** 2 + a.get("se", 0.0) ** 2) ** 0.5
        reg = delta < -2.0 * sigma
        row = {"base_acc": b.get("acc"), "adapter_acc": a.get("acc"),
               "delta": delta, "sigma": sigma, "regression": bool(reg)}
        if "ppl" in b and "ppl" in a:
            pdelta = a["ppl"] - b["ppl"]
            psigma = (b.get("ppl_se", 0.0) ** 2 + a.get("ppl_se", 0.0) ** 2) ** 0.5
            row["base_ppl"] = b["ppl"]
            row["adapter_ppl"] = a["ppl"]
            row["ppl_delta"] = pdelta
            row["ppl_sigma"] = psigma
            row["ppl_regression"] = bool(pdelta > 2.0 * psigma)
            reg = reg or row["ppl_regression"]
            row["regression"] = bool(reg)
        comparisons[task] = row
    return comparisons


# ensure lm_eval exists
try:
    import lm_eval  # noqa: F401
except Exception:
    print("installing lm-eval")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lm-eval"], check=True)

summary = {
    "experiment": "v032-bench",
    "kernel_id": KERNEL_ID,
    "kernel_revision": KERNEL_REVISION,
    "base_sha256": base_sha,
    "base_repo": BASE_REPO,
    "limit": LIMIT,
    "tasks": TASKS,
    "published_base_2k": BASE_2K,
    "runs": {},
    "comparisons": {},
}

base_ckpt = Path("/kaggle/working/base_v031.pt")
summary["runs"]["base"] = build_ckpt(None, base_ckpt)
base_results = run_bench(base_ckpt, Path("/kaggle/working/v032_base_2k.json"))
summary["runs"]["base"]["metrics"] = metrics(base_results)

for adapter_path in adapter_hits:
    seed = adapter_path.stem.replace("adapter_seed", "")
    ckpt = Path(f"/kaggle/working/v032_seed{seed}.pt")
    summary["runs"][seed] = build_ckpt(adapter_path, ckpt)
    results = run_bench(ckpt, Path(f"/kaggle/working/v032_seed{seed}_2k.json"))
    summary["runs"][seed]["metrics"] = metrics(results)
    summary["comparisons"][seed] = compare(
        summary["runs"]["base"]["metrics"], summary["runs"][seed]["metrics"]
    )

summary["V5"] = all(
    not any(row["regression"] for row in comp.values())
    for comp in summary["comparisons"].values()
)
summary["advance_v5"] = bool(summary["V5"])
Path("/kaggle/working/v032_bench_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("\nV032 2K VERDICT")
print("  V5:", summary["V5"], "| advance:", summary["advance_v5"])
for seed, comp in summary["comparisons"].items():
    print(f"  seed {seed}:")
    for task, row in comp.items():
        print(f"    {task}: base {row['base_acc']:.4f} -> {row['adapter_acc']:.4f} "
              f"delta {row['delta']:+.4f} ({row['delta']/max(row['sigma'],1e-9):+.2f}σ) "
              f"regression {row['regression']}")
print("  summary: /kaggle/working/v032_bench_summary.json")
