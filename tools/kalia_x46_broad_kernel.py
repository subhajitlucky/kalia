"""Kaggle GPU kernel: X46 broad-template compiler.

Trains the same rank-16 LoRA on 17 templates (X43's seven plus ten new broad
templates), with 1-, 2- and 3-op combinations. The four X44 test templates are
never seen. Evaluation is on the untouched X44 fresh benchmark.

No base weights are trained. The reader is frozen.
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
BENCH_SHA = "b69bcd6102e134a88a853b27c5783002df16cfc60e20d07f54f9f20e0a7d897a"
RANK = 16
ALPHA = 16

# --- code -------------------------------------------------------------------
hits = sorted(Path("/kaggle/input").rglob("broad_templates.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached or lacks broad_templates.py: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))
for req in ("broad_templates.py", "fresh_benchmark.py", "edit_compiler.py",
            "intervention_pairs.py", "eval_ip.py", "model.py"):
    assert (work / req).exists(), f"{req} missing from code dataset"
print("code:", code_root)

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
    train_compiler,
)
import broad_templates as bt  # noqa: E402
import fresh_benchmark  # noqa: E402

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


def make_compiler() -> tuple[GPT, int]:
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]["model"]))
    load = model.load_state_dict(ckpt["model"], strict=False)
    assert not load.missing_keys and not load.unexpected_keys
    n = apply_lora(model, RANK, ALPHA)
    model.to(device)
    return model, n


# --- X44 fresh benchmark -----------------------------------------------------
bench_dir = Path("/kaggle/working/x44-bench")
manifest = fresh_benchmark.write_benchmark(bench_dir, seed=2026, n_per_cell=400)
bench = bench_dir / "bench.jsonl"
bench_sha = hashlib.sha256(bench.read_bytes()).hexdigest()
assert bench_sha == BENCH_SHA, (bench_sha, BENCH_SHA)
records = [json.loads(line) for line in bench.read_text().splitlines() if line.strip()]
print("fresh bench:", bench_sha, "n", len(records), flush=True)

try:
    import tiktoken
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"],
                   check=True)
    import tiktoken

enc = tiktoken.get_encoding("gpt2")
out_dir = Path("/kaggle/working/x46")
out_dir.mkdir(parents=True, exist_ok=True)

summary = {
    "experiment": "X46-broad",
    "base_repo": BASE_REPO,
    "base_sha256": base_sha,
    "bench_sha256": bench_sha,
    "bench_manifest": manifest,
    "rank": RANK,
    "alpha": ALPHA,
    "device": torch.cuda.get_device_name(0),
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

# --- data -------------------------------------------------------------------
train = bt.build_train(n_per_combo=300, seed=46)
dev = bt.build_dev(n_per_combo=80, seed=999)
print("broad train:", len(train), "dev:", len(dev), flush=True)
print("templates:", len({r["template_id"] for r in train}), flush=True)

# --- train/eval -------------------------------------------------------------
for seed in (1401, 1402):
    tag = f"BROAD_{seed}"
    print(f"\n=== {tag} ({len(train)} train, 3000 steps) ===", flush=True)
    model, params = make_compiler()
    info = train_compiler(
        model, train, dev, enc, out_dir,
        seed=seed, steps=3000, batch_size=32, lr=2e-4, warmup=100,
        eval_every=300, eval_n=150, device=device, label="broad",
    )
    load_lora(model, info["path"])
    model.eval()
    gen = generate_all(model, records, enc, device, max_new=64)
    payload = score_generated(
        base, records, gen,
        out_dir / f"generated_{tag}.jsonl",
        out_dir / f"end2end_{tag}.json",
        out_dir / f"end2end_{tag}_rows.jsonl",
        device, batch_size=16, enc=enc, max_prefix_tokens=64,
    )
    summary["adapters"][tag] = {
        "seed": seed, "train": info, "trainable_params": params,
        "end2end": payload,
    }
    print(f"{tag}: compiler_em {payload['compiler']['exact_match']:.4f} "
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


arms = [summary["adapters"][f"BROAD_{s}"] for s in (1401, 1402)]
summary["bars"] = {
    "G-1_fresh_pair": all(cell(a["end2end"], "F-P") >= 0.45 for a in arms),
    "G-2_rename_swap": all(op(a["end2end"], "rename+swap") >= 0.45 for a in arms),
    "G-3_number_place": all(op(a["end2end"], "number+place") >= 0.40 for a in arms),
    "G-4_triple": all(cell(a["end2end"], "F-T") >= 0.30 for a in arms),
    "G-5_compiler_em": all(a["end2end"]["compiler"]["exact_match"] >= 0.25 for a in arms),
    "G-6_frozen": base_sha == BASE_SHA and all(
        not p.requires_grad for p in base.parameters()
    ) and all(a["end2end"]["compiler"]["fallback_rate"] < 0.01 for a in arms),
    "values": {
        f"BROAD_{a['seed']}": {
            "F-S": cell(a["end2end"], "F-S"),
            "F-P": cell(a["end2end"], "F-P"),
            "F-T": cell(a["end2end"], "F-T"),
            "number+place": op(a["end2end"], "number+place"),
            "rename+swap": op(a["end2end"], "rename+swap"),
            "compiler_em": a["end2end"]["compiler"]["exact_match"],
            "fallback_rate": a["end2end"]["compiler"]["fallback_rate"],
        }
        for a in arms
    },
}
summary["elapsed_hours"] = (time.time() - t0) / 3600
(out_dir / "x46_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)

print("\nX46 VERDICT")
for k, v in summary["bars"].items():
    if k != "values":
        print(f"  {k}: {v}")
for tag, row in summary["bars"]["values"].items():
    print(f"  {tag}: " + " ".join(f"{k}={v:.4f}" for k, v in row.items()))
print("  elapsed_hours:", round(summary["elapsed_hours"], 2))
print("  summary: /kaggle/working/x46/x46_summary.json")
