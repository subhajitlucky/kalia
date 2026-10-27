"""Kaggle GPU kernel: X37 — does the Pāṇinian rule grammar beat a matched MLP?

Registered as ``docs/preregistrations/2026-10-27-X37-sutra-micro.md``. Ten 30M
arms (base, matched control, rules, full, exception × seeds 1338/1339), 763
steps. The kernel trains the arms, measures the swap sensitivity on the 20
stories, and reports the rule usage and exception rate.

Outputs: ``out/x37-<arm>-s<seed>/`` per arm, the ten configs,
``x37_summary.json`` and ``run_manifest.json``.
"""

import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ARM = "sutra"  # staging pins this
KERNEL_ID = "unknown"  # staging pins, from the committed metadata
KERNEL_REVISION = "unknown"  # staging pins, from git HEAD

SEEDS = ("1338", "1339")
ARMS = ("base", "control", "rules", "full", "exception")
STEPS = 763
BASE_CONFIG = "configs/micro-muon.yaml"
PROMPT_TOKENS = 384
CONT_TOKENS = 128
S1_ALLOWANCE = 0.005
S2_FLOOR = 0.02
S3_ENTROPY = 2.0
S3_MAX_SHARE = 0.5
S3_EFFECTIVE = 8.0

SHARED = dict(
    sutra_rules=64,
    sutra_rank=8,
    sutra_general_hidden=1024,
    sutra_chunk=64,
    sutra_carry_dim=64,
    sutra_balance=0.01,
    sutra_top_k=1,
)
FLAGS = {
    "base": {},
    "control": {"sutra_control": True, "sutra_control_hidden": 1408},
    "rules": dict(SHARED, sutra=True, sutra_carry=False, sutra_scope=False),
    "full": dict(SHARED, sutra=True, sutra_carry=True, sutra_scope=True),
    "exception": dict(SHARED, sutra=True, sutra_carry=True, sutra_scope=True,
                      sutra_exception=True),
}
EXPECTED_PARAMS = {
    "base": 29921280,
    "control": 32575488,
    "rules": 32627718,
    "full": 32627718,
    "exception": 32651148,
}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_fingerprint(root) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix in (".py", ".yaml"):
            h.update(str(p.relative_to(root)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def write_json(path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def write_run_manifest(out_path, **fields) -> dict:
    fields.update(
        kernel_id=KERNEL_ID,
        kernel_revision=KERNEL_REVISION,
        recorded=datetime.now(timezone.utc).isoformat(),
    )
    write_json(out_path, fields)
    print("run manifest: " + json.dumps(fields, sort_keys=True))
    return fields


def entropy(usage: torch.Tensor) -> float:
    p = usage / usage.sum().clamp_min(1e-9)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum())


# --- code and data -----------------------------------------------------------
hits = sorted(glob.glob("/kaggle/input/**/train.py", recursive=True))
preferred = [p for p in hits if "/kalia-code-dev/" in p]
assert preferred, (
    "kalia-code-dev not attached or lacks train.py; other matches "
    "(do NOT trust them, they are stale bundles): " + str(hits)
)
code_root = os.path.dirname(preferred[0])
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("train.py", "ablate.py", "data.py", "model.py", BASE_CONFIG,
            "eval/ttt_probe_stories.json", "eval_entities.py"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"
print("code:", code_root, "| arm:", ARM)

data_hits = sorted(glob.glob("/kaggle/input/**/train.bin", recursive=True))
preferred_data = [p for p in data_hits if "/kalia-prep-v2b/" in p]
data_hits = preferred_data or data_hits
assert data_hits, "attach subhajitlucky/kalia-prep-v2b for train.bin"
data_dir = os.path.dirname(data_hits[0])
assert os.path.exists(os.path.join(data_dir, "val.bin")), "val.bin missing"
print("data:", data_dir)

# --- configs and parameter match --------------------------------------------
import yaml  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

base = yaml.safe_load(open(BASE_CONFIG))
base["model"]["qk_norm"] = True
base["model"]["logit_softcap"] = 0.0
assert base["train"]["max_steps"] == STEPS
assert base["train"]["eval_interval"] == 100

built = {}
for seed in SEEDS:
    for arm in ARMS:
        cfg = deepcopy(base)
        cfg["train"]["seed"] = int(seed)
        cfg["model"].update(FLAGS[arm])
        name = f"x37-{arm}-s{seed}"
        path = Path("configs") / f"{name}.yaml"
        yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
        params = GPT(GPTConfig(**cfg["model"])).num_params()
        assert params == EXPECTED_PARAMS[arm], (arm, params, EXPECTED_PARAMS[arm])
        built[name] = {
            "arm": arm,
            "seed": int(seed),
            "params": params,
            "flags": FLAGS[arm],
            "config_sha256": sha256_file(path),
        }
        print(f"built {name}: {params:,} params")

control_params = built["x37-control-s1338"]["params"]
full_params = built["x37-full-s1338"]["params"]
assert abs(control_params - full_params) / control_params < 0.01, (
    control_params, full_params
)
print("parameter match: control", control_params, "full", full_params)

write_run_manifest(
    "/kaggle/working/run_manifest.json",
    experiment=ARM,
    seeds=[int(s) for s in SEEDS],
    steps=STEPS,
    base_config=BASE_CONFIG,
    base_config_sha256=sha256_file(BASE_CONFIG),
    arms=built,
    bars={"S1_allowance": S1_ALLOWANCE, "S2_floor": S2_FLOOR,
          "S3_entropy": S3_ENTROPY, "S3_max_share": S3_MAX_SHARE,
          "S3_effective": S3_EFFECTIVE},
    code_snapshot=code_root,
    code_fingerprint=code_fingerprint(code_root),
    data_dir=data_dir,
    train_bin={"path": data_hits[0], "bytes": os.path.getsize(data_hits[0]),
               "sha256": sha256_file(data_hits[0])},
    val_bin={"path": os.path.join(data_dir, "val.bin"),
             "bytes": os.path.getsize(os.path.join(data_dir, "val.bin")),
             "sha256": sha256_file(os.path.join(data_dir, "val.bin"))},
)

# --- train the ten arms ------------------------------------------------------
order = ",".join(f"x37-{arm}-s{seed}" for seed in SEEDS for arm in ARMS)
print(f"\n===== X37 ({ARM}) — arms in order: {order} =====")
proc = subprocess.run(
    [sys.executable, "ablate.py", "--arms", order, "--steps", str(STEPS),
     "--data-dir", data_dir, "--out-root", "/kaggle/working/out"],
)
if proc.returncode != 0:
    sys.exit(proc.returncode)


# --- val, swap sensitivity, rule usage --------------------------------------
def run_final_val(run: str) -> float:
    path = Path("/kaggle/working/out") / run / "val_log.csv"
    rows = path.read_text().strip().splitlines()[1:]
    return float(rows[-1].split(",")[1])


import tiktoken  # noqa: E402
from data import TokenDataset  # noqa: E402
from eval_entities import extract_entities  # noqa: E402
from eval_reversibility import load_model  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = tiktoken.get_encoding("gpt2")
stories = json.loads(Path("eval/ttt_probe_stories.json").read_text())


def nll_of_continuation(model, prompt_ids: list[int], cont_ids: list[int]) -> float:
    seq = (prompt_ids + cont_ids)[:model.cfg.context_len]
    if len(seq) <= len(prompt_ids) + 1:
        return float("nan")
    x = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(seq[1:], dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(x, y)
    per = torch.nn.functional.cross_entropy(
        logits[0].view(-1, logits.size(-1)), y[0], reduction="none"
    )
    start = len(prompt_ids) - 1
    return float(per[start:].mean())


def swap_text(prompt: str) -> str:
    names = extract_entities(prompt)
    if len(names) < 2:
        return prompt
    a, b = names[0], names[1]
    out = re.sub(rf"\b{re.escape(a)}\b", "__KALIA_A__", prompt)
    out = re.sub(rf"\b{re.escape(b)}\b", a, out)
    return out.replace("__KALIA_A__", b)


swap_rows = {}
for seed in SEEDS:
    for arm in ARMS:
        run = f"x37-{arm}-s{seed}"
        ckpt = Path("/kaggle/working/out") / run / "ckpt.pt"
        if not ckpt.exists():
            print(f"{run}: no checkpoint")
            continue
        model = load_model(ckpt, device)
        model.eval()
        vals = []
        for i, story in enumerate(stories):
            ids = enc.encode_ordinary(story["prompt"])[:PROMPT_TOKENS]
            prompt_text = enc.decode(ids)
            cont = enc.encode_ordinary(story["reference_continuation"])[:CONT_TOKENS]
            if len(cont) < 8:
                continue
            orig = nll_of_continuation(model, ids, cont)
            swapped = enc.encode_ordinary(swap_text(prompt_text))[:PROMPT_TOKENS]
            swap = nll_of_continuation(model, swapped, cont)
            if math.isfinite(orig) and math.isfinite(swap):
                vals.append(swap - orig)
        swap_rows[run] = sum(vals) / len(vals) if vals else None
        print(f"{run}: swap sensitivity {swap_rows[run]} on {len(vals)} stories")

usage_rows = {}
val_ds = TokenDataset(os.path.join(data_dir, "val.bin"), base["model"]["context_len"])
gen = torch.Generator().manual_seed(1234)
x_val, y_val = val_ds.get_batch(16, device, gen)
for seed in SEEDS:
    for arm in ("rules", "full", "exception"):
        run = f"x37-{arm}-s{seed}"
        ckpt = Path("/kaggle/working/out") / run / "ckpt.pt"
        if not ckpt.exists():
            continue
        model = load_model(ckpt, device)
        model.eval()
        with torch.no_grad():
            model(x_val, y_val)
        usages = [b.mlp.last_usage for b in model.blocks if hasattr(b.mlp, "last_usage")]
        excs = [b.mlp.last_exc for b in model.blocks if getattr(b.mlp, "last_exc", None) is not None]
        usage = torch.stack(usages).mean(0)
        ent = entropy(usage)
        usage_rows[run] = {
            "entropy": ent,
            "max_share": float(usage.max()),
            "effective": float(math.exp(ent)),
            "exception": float(torch.stack(excs).mean()) if excs else None,
        }
        print(f"{run}: usage entropy {ent:.3f} max {usage.max():.3f} "
              f"effective {math.exp(ent):.1f} exception {usage_rows[run]['exception']}")

vals = {f"x37-{arm}-s{seed}": run_final_val(f"x37-{arm}-s{seed}")
        for seed in SEEDS for arm in ARMS}
control_val = sum(vals[f"x37-control-s{s}"] for s in SEEDS) / len(SEEDS)
full_val = sum(vals[f"x37-full-s{s}"] for s in SEEDS) / len(SEEDS)
rules_val = sum(vals[f"x37-rules-s{s}"] for s in SEEDS) / len(SEEDS)
control_swap = {s: swap_rows.get(f"x37-control-s{s}") for s in SEEDS}
full_swap = {s: swap_rows.get(f"x37-full-s{s}") for s in SEEDS}
rules_swap = {s: swap_rows.get(f"x37-rules-s{s}") for s in SEEDS}


def _delta(a, b):
    if a is None or b is None:
        return float("nan")
    return a - b


swap_deltas = {s: _delta(full_swap[s], control_swap[s]) for s in SEEDS}
control_spread = abs(vals["x37-control-s1338"] - vals["x37-control-s1339"])
swap_values = [v for v in control_swap.values() if v is not None]
swap_spread = (max(swap_values) - min(swap_values)) if len(swap_values) == len(SEEDS) else float("nan")
s1 = (full_val - control_val) <= S1_ALLOWANCE
s2 = (all(math.isfinite(d) for d in swap_deltas.values())
      and sum(swap_deltas.values()) / len(swap_deltas) >= max(swap_spread, S2_FLOOR)
      and all(d > 0 for d in swap_deltas.values()))
full_usage = [usage_rows[f"x37-full-s{s}"] for s in SEEDS if f"x37-full-s{s}" in usage_rows]
s3 = bool(full_usage) and all(
    u["entropy"] > S3_ENTROPY and u["max_share"] < S3_MAX_SHARE and u["effective"] > S3_EFFECTIVE
    for u in full_usage
)
s5 = {
    "full_minus_rules_val": full_val - rules_val,
    "full_minus_rules_swap": {
        s: _delta(full_swap[s], rules_swap[s]) for s in SEEDS
    },
}
summary = {
    "experiment": ARM,
    "seeds": [int(s) for s in SEEDS],
    "steps": STEPS,
    "vals": vals,
    "swap": swap_rows,
    "usage": usage_rows,
    "bars": {"S1_allowance": S1_ALLOWANCE, "S2_floor": S2_FLOOR,
             "S3_entropy": S3_ENTROPY, "S3_max_share": S3_MAX_SHARE,
             "S3_effective": S3_EFFECTIVE},
    "control_spread_val": control_spread,
    "control_spread_swap": swap_spread,
    "swap_deltas": swap_deltas,
    "S1": bool(s1),
    "S2": bool(s2),
    "S3": bool(s3),
    "S5": s5,
    "advance_full": bool(s1 and s2 and s3),
}
write_json("/kaggle/working/x37_summary.json", summary)
print("\nX37 VERDICT")
print(f"  S1 {s1} | S2 {s2} | S3 {s3} | advance full {summary['advance_full']}")
print(f"  val: base {vals[f'x37-base-s{SEEDS[0]}']:.4f} control {control_val:.4f} "
      f"rules {rules_val:.4f} full {full_val:.4f}")
print(f"  swap deltas: {swap_deltas} | control swap spread {swap_spread:.4f}")
print(f"  S5: {s5}")
print("  summary written to /kaggle/working/x37_summary.json")
