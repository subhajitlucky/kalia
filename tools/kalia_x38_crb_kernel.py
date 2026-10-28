"""Kaggle GPU kernel: X38 — the counterfactual residual bottleneck at 30M.

Registered as ``docs/preregistrations/2026-10-28-X38-crb-micro.md``. Four arms
(control, split-only, loss-only, CRB) × seeds 1338/1339, 763 steps each.

The mechanism: each block's residual delta is split into a low-rank stable
stream and a volatile complement. The sum is the standard residual, so the LM
path is invariant to the split; a paired counterfactual loss trains the stable
stream to ignore meaning-preserving edits and react to meaning-changing ones.

Outputs: the eight configs, ``out/x38-*`` checkpoints, ``x38_summary.json``,
``run_manifest.json`` and the pair audit.
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

ARM = "crb"  # staging pins this
KERNEL_ID = "unknown"  # staging pins, from the committed metadata
KERNEL_REVISION = "unknown"  # staging pins, from git HEAD

SEEDS = ("1338", "1339")
ARMS = ("control", "split", "loss", "crb")
STEPS = 763
BASE_CONFIG = "configs/micro-muon.yaml"
PROMPT_TOKENS = 384
CONT_TOKENS = 128
PARA_LEN = 128
C1_ALLOWANCE = 0.005
C2_FLOOR = 0.01
C3_MARGIN = 0.01
C4_ALLOWANCE = 0.005
C5_RANK = 8.0
C5_NORM = 0.01
C5_CHANGE = 1e-3

MODEL_FLAGS = {
    "control": {"crb": False},
    "split": {"crb": True, "crb_stable_dim": 64},
    "loss": {"crb": False},
    "crb": {"crb": True, "crb_stable_dim": 64},
}
CRB_WEIGHT = {"control": 0.0, "split": 0.0, "loss": 0.1, "crb": 0.1}
EXPECTED_PARAMS = {
    "control": 29921280,
    "split": 29970432,
    "loss": 29921280,
    "crb": 29970432,
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


# --- code and inputs ---------------------------------------------------------
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
for req in (
    "train.py", "model.py", "data.py", "crb_pairs.py", "eval_val.py",
    "eval_entities.py", "eval_reversibility.py", "configs/micro-muon.yaml",
    "eval/ttt_probe_stories.json",
):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"
print("code:", code_root)

data_hits = sorted(glob.glob("/kaggle/input/**/train.bin", recursive=True))
preferred_data = [p for p in data_hits if "/kalia-prep-v2b/" in p]
data_hits = preferred_data or data_hits
assert data_hits, "attach subhajitlucky/kalia-prep-v2b"
train_bin = data_hits[0]
data_dir = os.path.dirname(train_bin)
val_bin = os.path.join(data_dir, "val.bin")
assert os.path.exists(val_bin), "val.bin missing"
print("data:", data_dir)

# --- pairs -------------------------------------------------------------------
pairs_dir = Path("/kaggle/working/crb")
print("\n===== X38: generating counterfactual pairs =====")
subprocess.run(
    [sys.executable, "crb_pairs.py", "--out-dir", str(pairs_dir),
     "--seed", SEEDS[0], "--n-train", "1200", "--n-eval", "200"],
    check=True,
)
pair_audit = json.loads((pairs_dir / "audit.json").read_text())
assert not set(pair_audit["train_templates"]) & set(pair_audit["eval_templates"])
assert not set(pair_audit["train_names"]) & set(pair_audit["eval_names"])
print("pairs:", pair_audit["n_train"], "train /", pair_audit["n_eval"], "eval")

# --- configs and parameter match --------------------------------------------
import yaml  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

base = yaml.safe_load(open(BASE_CONFIG))
base["model"]["qk_norm"] = True
base["model"]["logit_softcap"] = 0.0
base["train"]["max_steps"] = STEPS
base["train"]["eval_interval"] = 100
base["train"]["log_interval"] = 10
base["train"]["crb_margin"] = 0.5
base["train"]["crb_batch_size"] = 8
base["train"]["crb_every"] = 1
base["train"]["crb_max_len"] = PARA_LEN

built = {}
for seed in SEEDS:
    for arm in ARMS:
        cfg = deepcopy(base)
        cfg["train"]["seed"] = int(seed)
        cfg["train"]["crb_weight"] = CRB_WEIGHT[arm]
        cfg["model"].update(MODEL_FLAGS[arm])
        name = f"x38-{arm}-s{seed}"
        path = Path("configs") / f"{name}.yaml"
        yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
        params = GPT(GPTConfig(**cfg["model"])).num_params()
        assert params == EXPECTED_PARAMS[arm], (arm, params, EXPECTED_PARAMS[arm])
        built[name] = {
            "arm": arm,
            "seed": int(seed),
            "params": params,
            "flags": MODEL_FLAGS[arm],
            "crb_weight": CRB_WEIGHT[arm],
            "config_sha256": sha256_file(path),
        }
        print(f"built {name}: {params:,} params")

assert EXPECTED_PARAMS["crb"] - EXPECTED_PARAMS["control"] == 49152
assert abs(EXPECTED_PARAMS["crb"] - EXPECTED_PARAMS["split"]) == 0
assert abs(EXPECTED_PARAMS["crb"] - EXPECTED_PARAMS["control"]) / EXPECTED_PARAMS["control"] < 0.01
print("parameter match: split/crb identical; crb vs control 0.17%")

# --- train -------------------------------------------------------------------
order = ",".join(f"x38-{arm}-s{seed}" for seed in SEEDS for arm in ARMS)
print(f"\n===== X38 ({ARM}) — arms in order: {order} =====")
for seed in SEEDS:
    for arm in ARMS:
        name = f"x38-{arm}-s{seed}"
        out_dir = Path("/kaggle/working/out") / name
        cmd = [
            sys.executable, "train.py",
            "--config", str(Path("configs") / f"{name}.yaml"),
            "--data-dir", data_dir,
            "--out-dir", str(out_dir),
        ]
        if arm in ("loss", "crb"):
            cmd += ["--cf-pairs", str(pairs_dir / "train.jsonl")]
        print("\n$", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            sys.exit(proc.returncode)

write_run_manifest(
    "/kaggle/working/run_manifest.json",
    experiment=ARM,
    seeds=[int(s) for s in SEEDS],
    steps=STEPS,
    base_config=BASE_CONFIG,
    base_config_sha256=sha256_file(BASE_CONFIG),
    arms=built,
    pairs={
        "dir": str(pairs_dir),
        "train_sha256": sha256_file(pairs_dir / "train.jsonl"),
        "eval_sha256": sha256_file(pairs_dir / "eval.jsonl"),
        "audit_sha256": sha256_file(pairs_dir / "audit.json"),
        "train_templates": pair_audit["train_templates"],
        "eval_templates": pair_audit["eval_templates"],
        "train_names": pair_audit["train_names"],
        "eval_names": pair_audit["eval_names"],
    },
    bars={"C1": C1_ALLOWANCE, "C2": C2_FLOOR, "C3": C3_MARGIN,
          "C4": C4_ALLOWANCE, "C5_rank": C5_RANK, "C5_norm": C5_NORM,
          "C5_change": C5_CHANGE},
    code_snapshot=code_root,
    code_fingerprint=code_fingerprint(code_root),
    data_dir=data_dir,
    train_bin={"path": train_bin, "bytes": os.path.getsize(train_bin),
               "sha256": sha256_file(train_bin)},
    val_bin={"path": val_bin, "bytes": os.path.getsize(val_bin),
             "sha256": sha256_file(val_bin)},
)

# --- evaluation --------------------------------------------------------------
import tiktoken  # noqa: E402
from data import TokenDataset  # noqa: E402
from eval_entities import extract_entities, retention  # noqa: E402
from eval_reversibility import load_model  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = tiktoken.get_encoding("gpt2")
stories = json.loads(Path("eval/ttt_probe_stories.json").read_text())


def run_final_val(name: str) -> float:
    out_dir = Path("/kaggle/working/out") / name
    ckpt = out_dir / "ckpt.pt"
    md = out_dir / "eval_val.md"
    subprocess.run(
        [sys.executable, "eval_val.py", "--ckpt", str(ckpt),
         "--val-bin", val_bin, "--batches", "100", "--batch-size", "8",
         "--seed", "1234", "--out", str(md), "--device", str(device)],
        check=True,
    )
    text = md.read_text()
    return float(re.search(r"Val loss:\s*([0-9.]+)", text).group(1))


def nll_of_continuation(model, prompt_ids: list[int], cont_ids: list[int]) -> float:
    seq = (prompt_ids + cont_ids)[: model.cfg.context_len]
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


def swap_sensitivity(model) -> float:
    vals = []
    for story in stories:
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
    return sum(vals) / len(vals) if vals else None


def encode_pair(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    ids = enc.encode_ordinary(text)[:PARA_LEN]
    mask = torch.zeros(PARA_LEN, dtype=torch.bool)
    mask[: len(ids)] = True
    out = torch.full((PARA_LEN,), 50256, dtype=torch.long)
    out[: len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out, mask


eval_rows = [json.loads(line) for line in (pairs_dir / "eval.jsonl").read_text().splitlines() if line.strip()]
eval_rows = [r for r in eval_rows if r["edit_type"] == "invariant"]
para_orig = torch.stack([encode_pair(r["original"])[0] for r in eval_rows]).to(device)
para_edit = torch.stack([encode_pair(r["edited"])[0] for r in eval_rows]).to(device)
para_mask0 = torch.stack([encode_pair(r["original"])[1] for r in eval_rows]).to(device)
para_mask1 = torch.stack([encode_pair(r["edited"])[1] for r in eval_rows]).to(device)


@torch.no_grad()
def paraphrase_distance(model) -> float:
    z0 = model.pooled_hidden(para_orig, para_mask0)
    z1 = model.pooled_hidden(para_edit, para_mask1)
    num = ((z0 - z1) ** 2).sum(dim=1)
    den = z0.pow(2).sum(dim=1) + z1.pow(2).sum(dim=1) + 1e-9
    return float((num / den).mean())


def greedy_retention(model) -> float:
    vals = []
    for story in stories:
        ids = enc.encode_ordinary(story["prompt"])[:PROMPT_TOKENS]
        if len(ids) < 8:
            continue
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(x, max_new_tokens=64, temperature=0.01, top_k=1)
        text = enc.decode(out[0, len(ids):].tolist())
        vals.append(retention(story["prompt"], text))
    return sum(vals) / len(vals) if vals else None


val_ds = TokenDataset(val_bin, 512)
gen = torch.Generator().manual_seed(1234)
x_val, _ = val_ds.get_batch(16, device, gen)

vals, swaps, paras, rets, stable, rel_change = {}, {}, {}, {}, {}, {}
for seed in SEEDS:
    for arm in ARMS:
        name = f"x38-{arm}-s{seed}"
        ckpt = Path("/kaggle/working/out") / name / "ckpt.pt"
        print(f"\n=== {name} ===")
        vals[name] = run_final_val(name)
        model = load_model(ckpt, device)
        model.eval()
        swaps[name] = swap_sensitivity(model)
        paras[name] = paraphrase_distance(model)
        rets[name] = greedy_retention(model)
        print(f"  val {vals[name]:.4f} | swap {swaps[name]:+.5f} | "
              f"paraphrase {paras[name]:.5f} | retention {rets[name]:.3f}")
        if arm in ("split", "crb"):
            stable[name] = model.stable_stats(x_val)
            cfg = yaml.safe_load(open(Path("configs") / f"{name}.yaml"))
            torch.manual_seed(int(seed))
            initial = GPT(GPTConfig(**cfg["model"]))
            rel = {}
            for key in ("up", "down"):
                a = getattr(initial.crb, key).weight.detach()
                b = getattr(model, key).weight.detach().cpu()
                rel[key] = float((b - a).norm() / (a.norm() + 1e-9))
            rel_change[name] = rel
            stable[name]["relative_change"] = rel
            print(f"  stable {stable[name]} | relative {rel}")


def delta(a, b):
    if a is None or b is None:
        return float("nan")
    return a - b


val_delta = {s: delta(vals[f"x38-crb-s{s}"], vals[f"x38-split-s{s}"]) for s in SEEDS}
val_guard = {s: delta(vals[f"x38-split-s{s}"], vals[f"x38-control-s{s}"]) for s in SEEDS}
swap_delta = {s: delta(swaps[f"x38-crb-s{s}"], swaps[f"x38-split-s{s}"]) for s in SEEDS}
para_delta = {s: delta(paras[f"x38-split-s{s}"], paras[f"x38-crb-s{s}"]) for s in SEEDS}
ret_delta = {s: delta(rets[f"x38-crb-s{s}"], rets[f"x38-split-s{s}"]) for s in SEEDS}

split_swap_spread = abs(swaps["x38-split-s1338"] - swaps["x38-split-s1339"])
c1 = all(math.isfinite(d) and d <= C1_ALLOWANCE for d in val_delta.values())
c1b = all(math.isfinite(d) and d <= C1_ALLOWANCE for d in val_guard.values())
c2_threshold = max(split_swap_spread, C2_FLOOR)
c2 = (all(math.isfinite(d) and d >= c2_threshold for d in swap_delta.values())
      and all(d > 0 for d in swap_delta.values()))
c3 = all(math.isfinite(d) and d >= C3_MARGIN for d in para_delta.values())
c4 = all(math.isfinite(d) and d >= -C4_ALLOWANCE for d in ret_delta.values())
c5 = all(
    stable.get(f"x38-crb-s{s}", {}).get("effective_rank", 0) > C5_RANK
    and stable.get(f"x38-crb-s{s}", {}).get("stable_norm", 0) > C5_NORM
    and min(stable.get(f"x38-crb-s{s}", {}).get("relative_change", {}).values() or [0]) > C5_CHANGE
    for s in SEEDS
)
c6 = not (set(pair_audit["train_templates"]) & set(pair_audit["eval_templates"])) and not (
    set(pair_audit["train_names"]) & set(pair_audit["eval_names"])
)
advance = bool(c1 and c2 and c3 and c5)

summary = {
    "experiment": ARM,
    "seeds": [int(s) for s in SEEDS],
    "steps": STEPS,
    "vals": vals,
    "swap": swaps,
    "paraphrase": paras,
    "retention": rets,
    "stable": stable,
    "params": {name: b["params"] for name, b in built.items()},
    "bars": {"C1": C1_ALLOWANCE, "C2": C2_FLOOR, "C3": C3_MARGIN,
             "C4": C4_ALLOWANCE, "C5_rank": C5_RANK, "C5_norm": C5_NORM,
             "C5_change": C5_CHANGE},
    "val_delta_crb_minus_split": val_delta,
    "val_guard_split_minus_control": val_guard,
    "swap_delta_crb_minus_split": swap_delta,
    "paraphrase_delta_split_minus_crb": para_delta,
    "retention_delta_crb_minus_split": ret_delta,
    "split_swap_spread": split_swap_spread,
    "C1": bool(c1),
    "C1b": bool(c1b),
    "C2": bool(c2),
    "C3": bool(c3),
    "C4": bool(c4),
    "C5": bool(c5),
    "C6": bool(c6),
    "advance": advance,
}
write_json("/kaggle/working/x38_summary.json", summary)
print("\nX38 VERDICT")
for key in ("C1", "C1b", "C2", "C3", "C4", "C5", "C6", "advance"):
    print(f"  {key}: {summary[key]}")
print(f"  val deltas (crb - split): {val_delta}")
print(f"  swap deltas (crb - split): {swap_delta} | threshold {c2_threshold:.5f}")
print(f"  paraphrase deltas (split - crb): {para_delta}")
print(f"  stable: {stable}")
print("  summary written to /kaggle/working/x38_summary.json")
