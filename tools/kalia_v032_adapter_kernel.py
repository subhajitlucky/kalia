"""Kaggle GPU kernel: V032 — frozen-base output-adapter fine-tune of v0.3.1.

Registered as ``docs/preregistrations/2026-11-04-V032-output-adapter.md``.
Known mechanism, engineering not novelty. Base v0.3.1 is frozen; only a
zero-init output adapter (512 -> 1024 -> 512) trains. Two seeds, 2,000 steps,
canonical 100-batch val. GitHub only; no Hugging Face push.
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

KERNEL_ID = "unknown"
KERNEL_REVISION = "unknown"
BASE_REPO = "kalia-lm/kalia-v031"
BASE_VAL_REF = 2.7579
SEEDS = ("1401", "1402")
STEPS = 2000
MICRO = 8
GRAD_ACCUM = 8
LR = 1e-3
WARMUP = 100
ADAPTER_HIDDEN = 1024
EVAL_BATCHES = 100
EVAL_BATCH = 8
EVAL_SEED = 1234
EVAL_EVERY = 200
EVAL_INTERIM = 20
BAR = 0.005

# --- code and data -----------------------------------------------------------
hits = sorted(Path("/kaggle/input").rglob("train.py"))
preferred = [p for p in hits if "/kalia-code-dev/" in str(p)]
assert preferred, "kalia-code-dev not attached or lacks train.py"
code_root = str(preferred[0].parent)
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("model.py", "data.py"):
    assert os.path.exists(req), f"{req} missing"
print("code:", code_root)

data_hits = sorted(Path("/kaggle/input").rglob("train.bin"))
preferred_data = [p for p in data_hits if "kalia-prep-v2b" in str(p)]
data_hits = preferred_data or data_hits
assert data_hits, "attach kalia-prep-v2b"
train_bin = str(data_hits[0])
data_dir = str(Path(train_bin).parent)
val_bin = os.path.join(data_dir, "val.bin")
assert os.path.exists(val_bin), "val.bin missing"
print("data:", data_dir)

# --- dependencies and base ---------------------------------------------------
try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "safetensors", "huggingface_hub"], check=True)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

from model import GPT, GPTConfig  # noqa: E402
from data import TokenDataset  # noqa: E402

base_dir = Path("/kaggle/working/base")
base_dir.mkdir(parents=True, exist_ok=True)
sf = hf_hub_download(BASE_REPO, "model.safetensors", local_dir=str(base_dir))
cf = hf_hub_download(BASE_REPO, "config.json", local_dir=str(base_dir))
base_sd = load_file(sf)
base_sha = hashlib.sha256(Path(sf).read_bytes()).hexdigest()
raw_cfg = json.loads(Path(cf).read_text())
cfg = {k: v for k, v in raw_cfg.items() if k in GPTConfig.__dataclass_fields__}
assert cfg["n_layer"] == 10 and cfg["n_embd"] == 512
print("base:", sf, "sha", base_sha[:16], "cfg", cfg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
base = GPT(GPTConfig(**cfg)).to(device)
r = base.load_state_dict(base_sd, strict=False)
assert not r.missing_keys and not r.unexpected_keys, (r.missing_keys, r.unexpected_keys)
base.eval()

adapter_cfg = dict(cfg)
adapter_cfg["memory_control"] = True
adapter_cfg["memory_control_hidden"] = ADAPTER_HIDDEN


def build_adapter():
    m = GPT(GPTConfig(**adapter_cfg)).to(device)
    load = m.load_state_dict(base_sd, strict=False)
    assert all(k.startswith("adapter.") for k in load.missing_keys), load.missing_keys
    assert not load.unexpected_keys, load.unexpected_keys
    for p in m.parameters():
        p.requires_grad = False
    for p in m.adapter.parameters():
        p.requires_grad = True
    return m


probe = build_adapter()
trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
assert trainable == 2 * cfg["n_embd"] * ADAPTER_HIDDEN, trainable
print("adapter trainable:", trainable)
del probe


def base_checksum(model):
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if name.startswith("adapter."):
            continue
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, ds, batches=EVAL_BATCHES, batch_size=EVAL_BATCH, seed=EVAL_SEED):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    total, tokens = 0.0, 0
    for _ in range(batches):
        x, y = ds.get_batch(batch_size, device, gen)
        _, loss = model(x, y)
        total += float(loss.item()) * int(y.numel())
        tokens += int(y.numel())
    return total / max(1, tokens)


val_ds = TokenDataset(val_bin, cfg["context_len"])
train_ds = TokenDataset(train_bin, cfg["context_len"])
base_val = evaluate(base)
adapter_zero = build_adapter()
zero_val = evaluate(adapter_zero)
print(f"base val {base_val:.4f} | zero-adapter val {zero_val:.4f} | ref {BASE_VAL_REF}")
assert abs(base_val - zero_val) < 1e-4, (base_val, zero_val)

summary = {
    "experiment": "v032-output-adapter",
    "kernel_id": KERNEL_ID,
    "kernel_revision": KERNEL_REVISION,
    "base_repo": BASE_REPO,
    "base_sha256": base_sha,
    "base_val": base_val,
    "base_val_ref": BASE_VAL_REF,
    "adapter_hidden": ADAPTER_HIDDEN,
    "adapter_params": trainable,
    "steps": STEPS,
    "micro": MICRO,
    "grad_accum": GRAD_ACCUM,
    "lr": LR,
    "seeds": [int(s) for s in SEEDS],
    "bar": BAR,
    "results": {},
}
train_log = Path("/kaggle/working/v032_train_log.csv")
val_log = Path("/kaggle/working/v032_val_log.csv")
train_log.write_text("seed,step,loss,lr,elapsed_s\n")
val_log.write_text("seed,step,val\n")


def run_seed(seed: str) -> dict:
    torch.manual_seed(int(seed))
    model = build_adapter()
    before = base_checksum(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.adapter.parameters() if p.requires_grad],
        lr=LR, betas=(0.9, 0.95), weight_decay=0.01,
    )
    warmup, total = WARMUP, STEPS
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup)
        * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / total)))),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    gen = torch.Generator().manual_seed(int(seed))
    best_val = float("inf")
    best_state = None
    start = time.time()
    for step in range(1, STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = train_ds.get_batch(MICRO, device, gen)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                _, loss = model(x, y)
            scaler.scale(loss / GRAD_ACCUM).backward()
            running += float(loss.item()) / GRAD_ACCUM
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.adapter.parameters() if p.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step % 50 == 0:
            with open(train_log, "a") as fh:
                fh.write(f"{seed},{step},{running:.4f},{optimizer.param_groups[0]['lr']:.6e},"
                         f"{time.time()-start:.1f}\n")
            print(f"seed {seed} step {step}/{STEPS} loss {running:.4f} "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}")
        if step % EVAL_EVERY == 0 or step == STEPS:
            val = evaluate(model, val_ds, batches=EVAL_INTERIM)
            with open(val_log, "a") as fh:
                fh.write(f"{seed},{step},{val:.4f}\n")
            print(f"seed {seed} step {step} val {val:.4f}")
            if val < best_val:
                best_val = val
                best_state = {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                    if k.startswith("adapter.")
                }
    assert best_state is not None
    model.load_state_dict(best_state, strict=False)
    final_val = evaluate(model, val_ds, batches=EVAL_BATCHES)
    after = base_checksum(model)
    assert before == after, "base parameters changed during adapter training"
    adapter_path = Path(f"/kaggle/working/adapter_seed{seed}.pt")
    torch.save({
        "adapter": best_state,
        "config": adapter_cfg,
        "base_sha256": base_sha,
        "base_val": base_val,
        "val": final_val,
        "seed": int(seed),
        "bar": BAR,
    }, adapter_path)
    print(f"seed {seed} FINAL val {final_val:.4f} | base {base_val:.4f} | "
          f"delta {final_val-base_val:+.4f} | frozen {before == after}")
    return {
        "val": final_val,
        "delta": final_val - base_val,
        "best_interim": best_val,
        "frozen_ok": before == after,
        "adapter_path": str(adapter_path),
        "adapter_bytes": adapter_path.stat().st_size,
    }


for seed in SEEDS:
    print(f"\n===== V032 adapter seed {seed} =====")
    summary["results"][seed] = run_seed(seed)

deltas = [r["delta"] for r in summary["results"].values()]
vals = [r["val"] for r in summary["results"].values()]
summary["min_delta"] = min(deltas)
summary["max_delta"] = max(deltas)
summary["all_frozen"] = all(r["frozen_ok"] for r in summary["results"].values())
summary["V1"] = all(d <= -BAR for d in deltas)
summary["V2"] = all(v < base_val for v in vals)
summary["V3"] = summary["all_frozen"]
summary["V4"] = summary["adapter_params"] == 2 * cfg["n_embd"] * ADAPTER_HIDDEN
summary["advance"] = bool(summary["V1"] and summary["V2"] and summary["V3"] and summary["V4"])
Path("/kaggle/working/v032_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("\nV032 VERDICT")
for key in ("V1", "V2", "V3", "V4", "advance"):
    print(f"  {key}: {summary[key]}")
print(f"  base val: {base_val:.4f} | adapter vals: {vals} | deltas: {deltas}")
print("  summary written to /kaggle/working/v032_summary.json")
