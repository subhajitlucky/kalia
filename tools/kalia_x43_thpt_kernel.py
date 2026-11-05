"""Kaggle GPU kernel: X43 template-held-out pair training.

Teaches all four test pair types on the three extra templates (disjoint from
the benchmark), then evaluates on the clean 2x2 benchmark. The compiler is the
same rank-16 LoRA as X41; the reader is frozen. No benchmark-template pair
supervision.

CPU/GPU artifacts only; the public model is unchanged.
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

BENCH_SEED = 2026
BENCH_SHA = "aad1b32187a8697b896f6be8408b979abd389466bd9fd42256b2f547e87dd45b"
BASE_REPO = "kalia-lm/kalia-v031"
BASE_SHA = "bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468"
RANK = 16
ALPHA = 16

hits = sorted(Path("/kaggle/input").rglob("edit_compiler.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached or lacks edit_compiler.py: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))
for req in ("edit_compiler.py", "composition_benchmark.py",
            "intervention_pairs.py", "eval_ip.py", "model.py"):
    assert (work / req).exists(), f"{req} missing from code dataset"
print("code:", code_root)
print("work:", work)

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
    build_dev,
    build_train,
    exact_match,
    generate_all,
    load_lora,
    score_generated,
    train_compiler,
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


def make_compiler() -> tuple[GPT, int]:
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]["model"]))
    load = model.load_state_dict(ckpt["model"], strict=False)
    assert not load.missing_keys and not load.unexpected_keys
    n = apply_lora(model, RANK, ALPHA)
    model.to(device)
    return model, n


bench_dir = Path("/kaggle/working/x43-bench")
if not bench_dir.exists():
    subprocess.run([sys.executable, "composition_benchmark.py",
                    "--out-dir", str(bench_dir), "--seed", str(BENCH_SEED)],
                   check=True)
bench = bench_dir / "bench.jsonl"
bench_sha = hashlib.sha256(bench.read_bytes()).hexdigest()
assert bench_sha == BENCH_SHA, (bench_sha, BENCH_SHA)
records = [json.loads(line) for line in bench.read_text().splitlines() if line.strip()]
print("bench sha256:", bench_sha, "records:", len(records))

try:
    import tiktoken
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"],
                   check=True)
    import tiktoken

enc = tiktoken.get_encoding("gpt2")
out_dir = Path("/kaggle/working/x43")
out_dir.mkdir(parents=True, exist_ok=True)

summary = {
    "experiment": "X43-thpt",
    "base_repo": BASE_REPO,
    "base_sha256": base_sha,
    "bench_sha256": bench_sha,
    "bench_seed": BENCH_SEED,
    "rank": RANK,
    "alpha": ALPHA,
    "device": torch.cuda.get_device_name(0),
    "arms": {},
    "bars": {},
}
t_start = time.time()

train_c = build_train("C", singles_per_op=1000, pairs_per_type=1000, seed=41)
dev_c = build_dev("C", singles_per_op=120, pairs_per_type=120, seed=999)
singles_dev = build_dev("A", singles_per_op=150, pairs_per_type=0, seed=777)
print("arm C train:", len(train_c), "dev:", len(dev_c), flush=True)

for seed in (1401, 1402):
    tag = f"C_{seed}"
    print(f"\n=== arm {tag} ({len(train_c)} train, 2200 steps) ===", flush=True)
    model, params = make_compiler()
    info = train_compiler(
        model, train_c, dev_c, enc, out_dir,
        seed=seed, steps=2200, batch_size=32, lr=2e-4, warmup=100,
        eval_every=300, eval_n=150, device=device, label="armC",
    )
    load_lora(model, info["path"])
    model.eval()
    single_gen = generate_all(model, singles_dev, enc, device, max_new=64)
    single_em = exact_match(singles_dev, single_gen)
    gen = generate_all(model, records, enc, device, max_new=64)
    payload = score_generated(
        base, records, gen,
        out_dir / f"generated_{tag}.jsonl",
        out_dir / f"end2end_{tag}.json",
        out_dir / f"end2end_{tag}_rows.jsonl",
        device, batch_size=16, enc=enc, max_prefix_tokens=64,
    )
    summary["arms"][tag] = {
        "seed": seed, "train": info, "trainable_params": params,
        "singles_dev": single_em, "end2end": payload,
    }
    print(f"arm {tag}: singles_em {single_em['exact_match']:.4f} "
          f"compiler_em {payload['compiler']['exact_match']:.4f} "
          f"fallback {payload['compiler']['fallback_rate']:.4f}", flush=True)
    print("  cells:", {k: round(v["accuracy"], 4)
                       for k, v in payload["by_cell"].items()}, flush=True)
    print("  ops:", {k: round(v["accuracy"], 4)
                     for k, v in sorted(payload["by_operation"].items())}, flush=True)
    del model
    torch.cuda.empty_cache()


def cell(arm, name):
    return arm["end2end"]["by_cell"][name]["accuracy"]


def op(arm, name):
    return arm["end2end"]["by_operation"][name]["accuracy"]


arms = [summary["arms"][f"C_{s}"] for s in (1401, 1402)]
summary["bars"] = {
    "D-1_singles_em": all(a["singles_dev"]["exact_match"] >= 0.90 for a in arms),
    "D-2_number_place": all(op(a, "number+place") >= 0.40 for a in arms),
    "D-3_rename_swap": all(op(a, "rename+swap") >= 0.45 for a in arms),
    "D-4_SP": all(cell(a, "S-P") >= 0.45 for a in arms),
    "D-5_UP": all(cell(a, "U-P") >= 0.35 for a in arms),
    "D-6_base_frozen": base_sha == BASE_SHA and all(
        not p.requires_grad for p in base.parameters()
    ),
    "values": {
        f"C_{a['seed']}": {
            "singles_em": a["singles_dev"]["exact_match"],
            "number+place": op(a, "number+place"),
            "rename+swap": op(a, "rename+swap"),
            "S-P": cell(a, "S-P"),
            "U-P": cell(a, "U-P"),
            "compiler_em": a["end2end"]["compiler"]["exact_match"],
            "fallback_rate": a["end2end"]["compiler"]["fallback_rate"],
        }
        for a in arms
    },
}
summary["elapsed_hours"] = (time.time() - t_start) / 3600
(out_dir / "x43_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)

print("\nX43 VERDICT")
for k, v in summary["bars"].items():
    if k != "values":
        print(f"  {k}: {v}")
for tag, row in summary["bars"]["values"].items():
    print(f"  {tag}: " + " ".join(f"{k}={v:.4f}" for k, v in row.items()))
print("  elapsed_hours:", round(summary["elapsed_hours"], 2))
print("  summary: /kaggle/working/x43/x43_summary.json")
