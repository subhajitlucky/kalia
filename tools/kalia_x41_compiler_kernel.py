"""Kaggle GPU kernel: X41 edit-compile-then-continue.

Stage 0: feasibility pilot (2k examples, 200 steps). Stop if dev single-edit
exact match < 0.50.
Stage 1: Arm B (held-out pair types) two seeds + Arm A (singles only) one seed.
Each adapter is evaluated end-to-end on the clean 2x2 benchmark: the compiler
writes the post-intervention prefix, the frozen v0.3.1 reader scores candidates.

No base weights are trained. The public model is unchanged.
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

# --- code -------------------------------------------------------------------
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
            "eval_composition_benchmark.py", "intervention_pairs.py",
            "eval_ip.py", "model.py"):
    assert (work / req).exists(), f"{req} missing from code dataset"
print("code:", code_root)
print("work:", work)

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
    build_dev,
    build_train,
    exact_match,
    generate_all,
    load_lora,
    score_bench,
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


def state_hash(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


reader_hash_before = state_hash(base)
base_ckpt = Path("/kaggle/working/base_v031.pt")
torch.save({"model": base.state_dict(), "config": {"model": cfg},
            "step": 0, "tokens": 0}, base_ckpt)
print("base sha256:", base_sha)
print("base params:", sum(p.numel() for p in base.parameters()))


def make_compiler() -> tuple[GPT, int]:
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]["model"]))
    load = model.load_state_dict(ckpt["model"], strict=False)
    assert not load.missing_keys and not load.unexpected_keys
    n = apply_lora(model, RANK, ALPHA)
    model.to(device)
    return model, n


# --- benchmark ---------------------------------------------------------------
bench_dir = Path("/kaggle/working/x41-bench")
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
    import tiktoken  # noqa: E402
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"],
                   check=True)
    import tiktoken  # noqa: E402

enc = tiktoken.get_encoding("gpt2")
out_dir = Path("/kaggle/working/x41")
out_dir.mkdir(parents=True, exist_ok=True)

summary = {
    "experiment": "X41-edit-compiler",
    "base_repo": BASE_REPO,
    "base_sha256": base_sha,
    "reader_hash_before": reader_hash_before,
    "bench_sha256": bench_sha,
    "bench_seed": BENCH_SEED,
    "rank": RANK,
    "alpha": ALPHA,
    "device": torch.cuda.get_device_name(0),
    "pilot": {},
    "arms": {},
    "baseline": {},
    "bars": {},
}
t_start = time.time()

# --- baseline (paired, same kernel) -----------------------------------------
print("\n=== baseline ===", flush=True)
for tag, mode in (("normal", "instruction_prefix"), ("gold", "edited_prefix")):
    payload = score_bench(
        base, bench, out_dir / f"baseline_{tag}.json",
        out_dir / f"baseline_{tag}_rows.jsonl", mode, device, batch_size=16,
    )
    summary["baseline"][tag] = payload
    print(f"baseline {tag}:", {k: round(v["accuracy"], 4)
                               for k, v in payload["by_cell"].items()}, flush=True)

# --- stage 0 pilot -----------------------------------------------------------
print("\n=== stage 0 pilot ===", flush=True)
pilot_train = build_train("B", singles_per_op=80, pairs_per_type=60, seed=41)
pilot_dev = build_dev("A", singles_per_op=60, seed=999)
print("pilot train:", len(pilot_train), "dev singles:", len(pilot_dev), flush=True)
pilot_model, pilot_params = make_compiler()
print("pilot trainable params:", pilot_params, flush=True)
pilot = train_compiler(
    pilot_model, pilot_train, pilot_dev, enc, out_dir,
    seed=1401, steps=200, batch_size=32, lr=2e-4, warmup=40,
    eval_every=100, eval_n=100, device=device, label="pilot",
)
summary["pilot"] = pilot
del pilot_model
torch.cuda.empty_cache()
if pilot["best_dev_em"] < 0.50:
    summary["stop"] = "stage0_pilot_below_0.50"
    (out_dir / "x41_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("STOP: pilot dev EM", pilot["best_dev_em"], "< 0.50", flush=True)
    raise SystemExit(0)

# --- stage 1 arms ------------------------------------------------------------
print("\n=== stage 1 ===", flush=True)
train_b = build_train("B", singles_per_op=1000, pairs_per_type=1000, seed=41)
dev_b = build_dev("B", singles_per_op=120, pairs_per_type=120, seed=999)
train_a = build_train("A", singles_per_op=1000, pairs_per_type=0, seed=41)
dev_a = build_dev("A", singles_per_op=120, pairs_per_type=0, seed=999)
singles_dev = build_dev("A", singles_per_op=150, pairs_per_type=0, seed=777)
print("arm B train:", len(train_b), "dev:", len(dev_b), flush=True)
print("arm A train:", len(train_a), "dev:", len(dev_a), flush=True)

ARMS = [
    ("B", 1401, train_b, dev_b, 2200),
    ("B", 1402, train_b, dev_b, 2200),
    ("A", 1401, train_a, dev_a, 1600),
]

for arm, seed, train_records, dev_records, steps in ARMS:
    tag = f"{arm}_{seed}"
    print(f"\n=== arm {tag} ({len(train_records)} train, {steps} steps) ===", flush=True)
    model, params = make_compiler()
    info = train_compiler(
        model, train_records, dev_records, enc, out_dir,
        seed=seed, steps=steps, batch_size=32, lr=2e-4, warmup=100,
        eval_every=300, eval_n=150, device=device, label=f"arm{arm}",
    )
    load_lora(model, info["path"])
    model.eval()
    # C-1: singles-only dev exact match on held-out names
    single_gen = generate_all(model, singles_dev, enc, device, max_new=64)
    single_em = exact_match(singles_dev, single_gen)
    # end-to-end test
    gen = generate_all(model, records, enc, device, max_new=64)
    payload = score_generated(
        base, records, gen,
        out_dir / f"generated_{tag}.jsonl",
        out_dir / f"end2end_{tag}.json",
        out_dir / f"end2end_{tag}_rows.jsonl",
        device, batch_size=16,
    )
    summary["arms"][tag] = {
        "arm": arm, "seed": seed, "train": info,
        "trainable_params": params,
        "singles_dev": single_em,
        "end2end": payload,
    }
    print(f"arm {tag}: singles_em {single_em['exact_match']:.4f} "
          f"compiler_em {payload['compiler']['exact_match']:.4f}", flush=True)
    print("  cells:", {k: round(v["accuracy"], 4)
                       for k, v in payload["by_cell"].items()}, flush=True)
    print("  ops:", {k: round(v["accuracy"], 4)
                     for k, v in sorted(payload["by_operation"].items())}, flush=True)
    del model
    torch.cuda.empty_cache()

# --- bars --------------------------------------------------------------------
def get_cell(arm, cell):
    return arm["end2end"]["by_cell"][cell]["accuracy"]


def get_op(arm, op):
    return arm["end2end"]["by_operation"][op]["accuracy"]


b_arms = [summary["arms"][f"B_{s}"] for s in (1401, 1402)]
summary["bars"] = {
    "C-1_singles_em": all(a["singles_dev"]["exact_match"] >= 0.90 for a in b_arms),
    "C-2_number_place": all(get_op(a, "number+place") >= 0.40 for a in b_arms),
    "C-3_rename_swap": all(get_op(a, "rename+swap") >= 0.45 for a in b_arms),
    "C-4_SP": all(get_cell(a, "S-P") >= 0.40 for a in b_arms),
    "C-5_UP": all(get_cell(a, "U-P") >= 0.30 for a in b_arms),
    "C-6_base_frozen": (
        reader_hash_before == state_hash(base)
        and all(not p.requires_grad for p in base.parameters())
        and base_sha == BASE_SHA
    ),
    "values": {
        f"{a['arm']}_{a['seed']}": {
            "singles_em": a["singles_dev"]["exact_match"],
            "number+place": get_op(a, "number+place"),
            "rename+swap": get_op(a, "rename+swap"),
            "S-P": get_cell(a, "S-P"),
            "U-P": get_cell(a, "U-P"),
        }
        for a in summary["arms"].values()
    },
}
summary["elapsed_hours"] = (time.time() - t_start) / 3600
(out_dir / "x41_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)

print("\nX41 VERDICT")
for k, v in summary["bars"].items():
    if k != "values":
        print(f"  {k}: {v}")
for tag, row in summary["bars"]["values"].items():
    print(f"  {tag}: " + " ".join(f"{k}={v:.4f}" for k, v in row.items()))
print("  elapsed_hours:", round(summary["elapsed_hours"], 2))
print("  summary: /kaggle/working/x41/x41_summary.json")
