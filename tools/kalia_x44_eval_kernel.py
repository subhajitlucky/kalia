"""Kaggle GPU kernel: X44 fresh benchmark + X45 depth.

Generates the fresh benchmark (new names/templates), scores v0.3.1 normal and
gold-edit, then scores the two X43 compiler adapters end-to-end. The F-T cell
is 8-way and tests zero-shot three-operation composition.

No training in this kernel. The base reader is frozen.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_REPO = "kalia-lm/kalia-v031"
BASE_SHA = "bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468"
RANK = 16
ALPHA = 16
SEEDS = ("1401", "1402")

# --- code -------------------------------------------------------------------
hits = sorted(Path("/kaggle/input").rglob("fresh_benchmark.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached or lacks fresh_benchmark.py: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))
for req in ("fresh_benchmark.py", "edit_compiler.py", "intervention_pairs.py",
            "eval_ip.py", "model.py"):
    assert (work / req).exists(), f"{req} missing from code dataset"
print("code:", code_root)

# --- adapters ---------------------------------------------------------------
adapter_hits = sorted(Path("/kaggle/input").rglob("lora_armC_seed*.pt"))
assert adapter_hits, "kalia-x43-adapters not attached"
print("adapters:", [str(p) for p in adapter_hits])

# --- base --------------------------------------------------------------------
try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "safetensors", "huggingface_hub", "tiktoken"], check=True)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

import torch  # noqa: E402

from model import GPT, GPTConfig  # noqa: E402
from edit_compiler import (  # noqa: E402
    apply_lora,
    generate_all,
    load_lora,
    score_bench,
    score_generated,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "this kernel needs a GPU"
print("device:", torch.cuda.get_device_name(0))
torch.set_float32_matmul_precision("high")

base_dir = Path("/kaggle/working/base")
base_dir.mkdir(parents=True, exist_ok=True)
sf = hf_hub_download(BASE_REPO, "model.safetensors", local_dir=str(base_dir))
cf = hf_hub_download(BASE_REPO, "config.json", local_dir=str(base_dir))
base_sha = hashlib.sha256(Path(sf).read_bytes()).hexdigest()
assert base_sha == BASE_SHA, (base_sha, BASE_SHA)
raw_cfg = json.loads(Path(cf).read_text())
cfg = {k: v for k, v in raw_cfg.items() if k in GPTConfig.__dataclass_fields__}
base = GPT(GPTConfig(**cfg))
load = base.load_state_dict(load_file(sf), strict=False)
assert not load.missing_keys and not load.unexpected_keys
base.to(device).eval()
for p in base.parameters():
    p.requires_grad = False
base_ckpt = Path("/kaggle/working/base_v031.pt")
torch.save({"model": base.state_dict(), "config": {"model": cfg},
            "step": 0, "tokens": 0}, base_ckpt)
print("base sha256:", base_sha)


def make_compiler(adapter_path: Path) -> GPT:
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]["model"]))
    load = model.load_state_dict(ckpt["model"], strict=False)
    assert not load.missing_keys and not load.unexpected_keys
    apply_lora(model, RANK, ALPHA)
    load_lora(model, adapter_path)
    model.to(device).eval()
    return model


# --- benchmark ---------------------------------------------------------------
try:
    import tiktoken
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"],
                   check=True)
    import tiktoken

import fresh_benchmark  # noqa: E402

enc = tiktoken.get_encoding("gpt2")
bench_dir = Path("/kaggle/working/x44-bench")
manifest = fresh_benchmark.write_benchmark(bench_dir, seed=2026, n_per_cell=400)
bench = bench_dir / "bench.jsonl"
bench_sha = hashlib.sha256(bench.read_bytes()).hexdigest()
records = [json.loads(line) for line in bench.read_text().splitlines() if line.strip()]
print("fresh bench:", bench_sha, "n", len(records), flush=True)
print("cells:", {k: v["n"] for k, v in manifest["cells"].items()}, flush=True)

out_dir = Path("/kaggle/working/x44")
out_dir.mkdir(parents=True, exist_ok=True)
summary = {
    "experiment": "X44-fresh-depth",
    "base_repo": BASE_REPO,
    "base_sha256": base_sha,
    "bench_sha256": bench_sha,
    "bench_manifest": manifest,
    "baseline": {},
    "adapters": {},
    "bars": {},
}
t0 = time.time()

# --- baseline ---------------------------------------------------------------
for tag, mode in (("normal", "instruction_prefix"), ("gold", "edited_prefix")):
    payload = score_bench(
        base, bench, out_dir / f"baseline_{tag}.json",
        out_dir / f"baseline_{tag}_rows.jsonl", mode, device, batch_size=16,
    )
    summary["baseline"][tag] = payload
    print(f"baseline {tag}:", {k: round(v["accuracy"], 4)
                               for k, v in payload["by_cell"].items()}, flush=True)

# --- adapters ---------------------------------------------------------------
for seed in SEEDS:
    adapter_path = next(p for p in adapter_hits if f"seed{seed}" in p.name)
    print(f"\n=== adapter {seed}: {adapter_path} ===", flush=True)
    model = make_compiler(adapter_path)
    gen = generate_all(model, records, enc, device, max_new=64)
    payload = score_generated(
        base, records, gen,
        out_dir / f"generated_C_{seed}.jsonl",
        out_dir / f"end2end_C_{seed}.json",
        out_dir / f"end2end_C_{seed}_rows.jsonl",
        device, batch_size=16, enc=enc, max_prefix_tokens=64,
    )
    summary["adapters"][seed] = payload
    print(f"adapter {seed}: compiler_em {payload['compiler']['exact_match']:.4f} "
          f"fallback {payload['compiler']['fallback_rate']:.4f}", flush=True)
    print("  cells:", {k: round(v["accuracy"], 4)
                       for k, v in payload["by_cell"].items()}, flush=True)
    print("  ops:", {k: round(v["accuracy"], 4)
                     for k, v in sorted(payload["by_operation"].items())}, flush=True)
    del model
    torch.cuda.empty_cache()


def cell(payload, name):
    return payload["by_cell"].get(name, {}).get("accuracy", 0.0)


def op(payload, name):
    return payload["by_operation"].get(name, {}).get("accuracy", 0.0)


arms = [summary["adapters"][s] for s in SEEDS]
summary["bars"] = {
    "F-1_fresh_pair": all(cell(a, "F-P") >= 0.40 for a in arms),
    "F-2_number_place": all(op(a, "number+place") >= 0.40 for a in arms),
    "F-3_rename_swap": all(op(a, "rename+swap") >= 0.45 for a in arms),
    "F-4_fallback": all(a["compiler"]["fallback_rate"] < 0.01 for a in arms),
    "F-5_base_frozen": base_sha == BASE_SHA and all(
        not p.requires_grad for p in base.parameters()
    ),
    "F-6_triple": all(cell(a, "F-T") >= 0.30 for a in arms),
    "values": {
        s: {
            "F-S": cell(summary["adapters"][s], "F-S"),
            "F-P": cell(summary["adapters"][s], "F-P"),
            "F-T": cell(summary["adapters"][s], "F-T"),
            "number+place": op(summary["adapters"][s], "number+place"),
            "rename+swap": op(summary["adapters"][s], "rename+swap"),
            "compiler_em": summary["adapters"][s]["compiler"]["exact_match"],
            "fallback_rate": summary["adapters"][s]["compiler"]["fallback_rate"],
        }
        for s in SEEDS
    },
}
summary["elapsed_hours"] = (time.time() - t0) / 3600
(out_dir / "x44_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)

print("\nX44/X45 VERDICT")
for k, v in summary["bars"].items():
    if k != "values":
        print(f"  {k}: {v}")
for seed, row in summary["bars"]["values"].items():
    print(f"  C_{seed}: " + " ".join(f"{k}={v:.4f}" for k, v in row.items()))
print("  elapsed_hours:", round(summary["elapsed_hours"], 2))
print("  summary: /kaggle/working/x44/x44_summary.json")
