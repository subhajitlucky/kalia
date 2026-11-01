"""Kaggle GPU kernel: X40 — intervention pretraining at 30M.

Registered as ``docs/preregistrations/2026-11-01-X40-intervention-pretraining.md``.
Five arms (control, adapter, pairs, workspace, IP) x seeds 1338/1339, 763
steps each.

The mechanism: an auxiliary intervention next-state loss carried by a small,
causal event workspace. The `pairs` arm is the data-augmentation baseline; the
`workspace` arm is the architecture control; only `IP` has the state head and
the diversity loss. The key bar is I-3: IP must beat `pairs` on **unseen
compositions**, not just on the interventions it trained on.

Outputs: the ten configs, ``out/x40-*`` checkpoints, ``x40_summary.json``,
``x40_seen_unseen_rows.jsonl``, ``run_manifest.json`` and the pair audit.
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

ARM = "ip"
KERNEL_ID = "unknown"  # staging pins, from the committed metadata
KERNEL_REVISION = "unknown"  # staging pins, from git HEAD

SEEDS = ("1338", "1339")
ARMS = ("control", "adapter", "pairs", "workspace", "ip")
PAIR_ARMS = ("pairs", "workspace", "ip")
STATE_ARMS = ("ip",)
STEPS = 763
BASE_CONFIG = "configs/micro-muon.yaml"
PREFIX_LEN = 64
SUFFIX_LEN = 64
IP_BATCH = 8
IP_LM_WEIGHT = 1.0
IP_STATE_WEIGHT = 0.1
IP_DIV_WEIGHT = 0.01
EVENT_SLOTS = 16
EVENT_SLOT_DIM = 64
EVENT_CHUNK = 64
EXPECTED_CONTROL = 29_921_280
I1_ALLOWANCE = 0.005
I2_MARGIN = 0.10
I3_MARGIN = 0.10
I4_ALLOWANCE = 0.005
I5_ENTROPY = 2.0
I5_RANK = 4.0
I5_PROBE = 0.10
PROMPT_TOKENS = 384
CONT_TOKENS = 128


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
    "train.py", "model.py", "data.py", "intervention_pairs.py", "eval_ip.py",
    "eval_val.py", "eval_entities.py", "eval_reversibility.py",
    "configs/micro-muon.yaml", "eval/ttt_probe_stories.json",
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
pairs_dir = Path("/kaggle/working/ip")
print("\n===== X40: generating intervention pairs =====")
subprocess.run(
    [sys.executable, "intervention_pairs.py", "--out-dir", str(pairs_dir),
     "--seed", SEEDS[0], "--n-train", "4000", "--n-eval", "150",
     "--n-comp", "150"],
    check=True,
)
pair_audit = json.loads((pairs_dir / "audit.json").read_text())
assert not set(pair_audit["train_templates"]) & set(pair_audit["eval_templates"])
assert not set(pair_audit["train_names"]) & set(pair_audit["eval_names"])
assert pair_audit["n_train"] == 16_000, pair_audit["n_train"]
assert pair_audit["n_eval"] == 1_050, pair_audit["n_eval"]
assert set(pair_audit["eval_counts"]) >= {
    "rename", "swap", "negate", "number", "place",
    "rename+swap", "swap+negate", "number+place", "rename+negate",
}
print("pairs:", pair_audit["n_train"], "train /", pair_audit["n_eval"], "eval")
print("lengths:", pair_audit["lengths"])

# --- configs and parameter match --------------------------------------------
import yaml  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

base = yaml.safe_load(open(BASE_CONFIG))
base["model"]["qk_norm"] = True
base["model"]["logit_softcap"] = 0.0  # D55: new configs do not carry the soft-cap
base["train"]["max_steps"] = STEPS
base["train"]["eval_interval"] = 100
base["train"]["log_interval"] = 10
base["train"]["ip_lm_weight"] = IP_LM_WEIGHT
base["train"]["ip_state_weight"] = IP_STATE_WEIGHT
base["train"]["ip_div_weight"] = IP_DIV_WEIGHT
base["train"]["ip_batch_size"] = IP_BATCH
base["train"]["ip_prefix_len"] = PREFIX_LEN
base["train"]["ip_suffix_len"] = SUFFIX_LEN
base["train"]["ip_every"] = 1

probe_cfg = deepcopy(base["model"])
probe_cfg.update({
    "event_slots": EVENT_SLOTS, "event_slot_dim": EVENT_SLOT_DIM,
    "event_chunk": EVENT_CHUNK, "event_head": False,
})
probe_model = GPT(GPTConfig(**probe_cfg))
event_params = sum(p.numel() for p in probe_model.event.parameters())
adapter_hidden = max(1, round(event_params / (2 * base["model"]["n_embd"])))
print(f"event params: {event_params:,} | adapter hidden: {adapter_hidden}")

MODEL_FLAGS = {
    "control": {},
    "adapter": {"memory_control": True, "memory_control_hidden": adapter_hidden},
    "pairs": {},
    "workspace": {
        "event_slots": EVENT_SLOTS, "event_slot_dim": EVENT_SLOT_DIM,
        "event_chunk": EVENT_CHUNK, "event_head": False,
    },
    "ip": {
        "event_slots": EVENT_SLOTS, "event_slot_dim": EVENT_SLOT_DIM,
        "event_chunk": EVENT_CHUNK, "event_head": True,
        "event_head_hidden": 256, "event_op_dim": 6,
    },
}
WEIGHTS = {
    "control": {"ip_lm_weight": 0.0, "ip_state_weight": 0.0, "ip_div_weight": 0.0},
    "adapter": {"ip_lm_weight": 0.0, "ip_state_weight": 0.0, "ip_div_weight": 0.0},
    "pairs": {"ip_lm_weight": IP_LM_WEIGHT, "ip_state_weight": 0.0, "ip_div_weight": 0.0},
    "workspace": {"ip_lm_weight": IP_LM_WEIGHT, "ip_state_weight": 0.0, "ip_div_weight": 0.0},
    "ip": {
        "ip_lm_weight": IP_LM_WEIGHT,
        "ip_state_weight": IP_STATE_WEIGHT,
        "ip_div_weight": IP_DIV_WEIGHT,
    },
}

built = {}
params = {}
for seed in SEEDS:
    for arm in ARMS:
        cfg = deepcopy(base)
        cfg["train"]["seed"] = int(seed)
        cfg["train"].update(WEIGHTS[arm])
        cfg["model"].update(MODEL_FLAGS[arm])
        name = f"x40-{arm}-s{seed}"
        path = Path("configs") / f"{name}.yaml"
        yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
        model = GPT(GPTConfig(**cfg["model"]))
        params[name] = model.num_params()
        built[name] = {
            "arm": arm,
            "seed": int(seed),
            "params": params[name],
            "flags": MODEL_FLAGS[arm],
            "weights": WEIGHTS[arm],
            "config_sha256": sha256_file(path),
        }
        print(f"built {name}: {params[name]:,} params")

for name, count in params.items():
    if name.startswith("x40-control-"):
        assert count == EXPECTED_CONTROL, (name, count, EXPECTED_CONTROL)
    if name.startswith("x40-pairs-"):
        assert count == EXPECTED_CONTROL, (name, count, EXPECTED_CONTROL)
for seed in SEEDS:
    control = params[f"x40-control-s{seed}"]
    adapter = params[f"x40-adapter-s{seed}"]
    workspace = params[f"x40-workspace-s{seed}"]
    ip = params[f"x40-ip-s{seed}"]
    assert abs(adapter - workspace) <= 2 * base["model"]["n_embd"], (
        adapter, workspace
    )
    assert workspace == control + event_params, (workspace, control, event_params)
    assert ip > workspace, (ip, workspace)
    print(
        f"seed {seed}: control {control:,} | adapter {adapter:,} | "
        f"workspace {workspace:,} | IP {ip:,} | "
        f"adapter vs workspace {abs(adapter - workspace):,}"
    )

# --- train -------------------------------------------------------------------
order = ",".join(f"x40-{arm}-s{seed}" for seed in SEEDS for arm in ARMS)
print(f"\n===== X40 ({ARM}) — arms in order: {order} =====")
for seed in SEEDS:
    for arm in ARMS:
        name = f"x40-{arm}-s{seed}"
        out_dir = Path("/kaggle/working/out") / name
        cmd = [
            sys.executable, "train.py",
            "--config", str(Path("configs") / f"{name}.yaml"),
            "--data-dir", data_dir,
            "--out-dir", str(out_dir),
        ]
        if arm in PAIR_ARMS:
            cmd += ["--ip-pairs", str(pairs_dir / "train.jsonl")]
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
        "lengths": pair_audit["lengths"],
    },
    bars={
        "I1": I1_ALLOWANCE, "I2": I2_MARGIN, "I3": I3_MARGIN,
        "I4": I4_ALLOWANCE, "I5_entropy": I5_ENTROPY,
        "I5_rank": I5_RANK, "I5_probe": I5_PROBE,
    },
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
from eval_ip import summarise_arm  # noqa: E402
from eval_reversibility import load_model  # noqa: E402
from intervention_pairs import InterventionDataset  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = tiktoken.get_encoding("gpt2")
stories = json.loads(Path("eval/ttt_probe_stories.json").read_text())
eval_ds = InterventionDataset(
    pairs_dir / "eval.jsonl", prefix_len=PREFIX_LEN, suffix_len=SUFFIX_LEN
)


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


vals, swaps, rets = {}, {}, {}
seen, unseen, overall, by_op, workspace = {}, {}, {}, {}, {}
rows_path = Path("/kaggle/working/x40_seen_unseen_rows.jsonl")
with open(rows_path, "w") as rows_fh:
    for seed in SEEDS:
        for arm in ARMS:
            name = f"x40-{arm}-s{seed}"
            ckpt = Path("/kaggle/working/out") / name / "ckpt.pt"
            print(f"\n=== {name} ===")
            vals[name] = run_final_val(name)
            model = load_model(ckpt, device)
            result = summarise_arm(model, eval_ds, device)
            accuracy = result["accuracy"]
            seen[name] = accuracy["seen"]
            unseen[name] = accuracy["unseen"]
            overall[name] = accuracy["overall"]
            by_op[name] = accuracy["by_operation"]
            workspace[name] = result["workspace"]
            for row in accuracy["rows"]:
                rows_fh.write(json.dumps({"model": name, **row}, sort_keys=True) + "\n")
            rets[name] = greedy_retention(model)
            swaps[name] = swap_sensitivity(model)
            print(
                f"  val {vals[name]:.4f} | seen {seen[name]:.4f} | "
                f"unseen {unseen[name]:.4f} | retention {rets[name]:.3f}"
            )
            if workspace[name]:
                w = workspace[name]
                print(
                    f"  workspace: entropy {w['slot_entropy_bits']:.3f} bits | "
                    f"rank {w['effective_rank']:.3f} | "
                    f"probe {w['probe_accuracy']:.3f} (+{w['probe_gain']:.3f})"
                )


def delta(a, b):
    if a is None or b is None:
        return float("nan")
    return a - b


val_delta = {s: delta(vals[f"x40-ip-s{s}"], vals[f"x40-control-s{s}"]) for s in SEEDS}
seen_delta = {s: delta(seen[f"x40-ip-s{s}"], seen[f"x40-control-s{s}"]) for s in SEEDS}
unseen_delta = {s: delta(unseen[f"x40-ip-s{s}"], unseen[f"x40-pairs-s{s}"]) for s in SEEDS}
ret_delta = {s: delta(rets[f"x40-ip-s{s}"], rets[f"x40-control-s{s}"]) for s in SEEDS}
pairs_unseen_delta = {
    s: delta(unseen[f"x40-pairs-s{s}"], unseen[f"x40-control-s{s}"]) for s in SEEDS
}
workspace_delta = {
    s: delta(unseen[f"x40-workspace-s{s}"], unseen[f"x40-pairs-s{s}"]) for s in SEEDS
}

i1 = all(math.isfinite(d) and d <= I1_ALLOWANCE for d in val_delta.values())
i2 = all(math.isfinite(d) and d >= I2_MARGIN for d in seen_delta.values())
i3 = all(math.isfinite(d) and d >= I3_MARGIN for d in unseen_delta.values())
i4 = all(math.isfinite(d) and d >= -I4_ALLOWANCE for d in ret_delta.values())
i5 = all(
    workspace.get(f"x40-ip-s{s}", {}).get("slot_entropy_bits", 0) > I5_ENTROPY
    and workspace.get(f"x40-ip-s{s}", {}).get("effective_rank", 0) > I5_RANK
    and workspace.get(f"x40-ip-s{s}", {}).get("probe_gain", 0) > I5_PROBE
    for s in SEEDS
)
i6 = not (set(pair_audit["train_templates"]) & set(pair_audit["eval_templates"])) and not (
    set(pair_audit["train_names"]) & set(pair_audit["eval_names"])
)
ruler = {
    metric: abs(values[f"x40-control-s{SEEDS[0]}"] - values[f"x40-control-s{SEEDS[1]}"])
    for metric, values in (("val", vals), ("seen", seen), ("unseen", unseen))
}
i7 = all(math.isfinite(v) for v in ruler.values())
advance = bool(i1 and i2 and i3 and i5)

if not i1:
    decision = "stop: the IP arm costs language quality (I-1)"
elif not i5:
    decision = "fix: the workspace collapsed (I-5); do not read I-3 as a mechanism verdict"
elif not i3:
    decision = "stop: no unseen-composition gain over pairs-only (I-3)"
elif not i2:
    decision = "stop: seen interventions were not learned (I-2)"
else:
    decision = "advance to the 58M confirmation"

summary = {
    "experiment": "x40-ip",
    "seeds": [int(s) for s in SEEDS],
    "steps": STEPS,
    "arms": built,
    "params": params,
    "pairs": {
        "n_train": pair_audit["n_train"],
        "n_eval": pair_audit["n_eval"],
        "train_counts": pair_audit["train_counts"],
        "eval_counts": pair_audit["eval_counts"],
        "lengths": pair_audit["lengths"],
        "train_sha256": sha256_file(pairs_dir / "train.jsonl"),
        "eval_sha256": sha256_file(pairs_dir / "eval.jsonl"),
        "audit_sha256": sha256_file(pairs_dir / "audit.json"),
    },
    "vals": vals,
    "seen": seen,
    "unseen": unseen,
    "overall": overall,
    "retention": rets,
    "swap": swaps,
    "workspace": workspace,
    "by_operation": by_op,
    "val_delta_ip_minus_control": val_delta,
    "seen_delta_ip_minus_control": seen_delta,
    "unseen_delta_ip_minus_pairs": unseen_delta,
    "retention_delta_ip_minus_control": ret_delta,
    "unseen_delta_pairs_minus_control": pairs_unseen_delta,
    "unseen_delta_workspace_minus_pairs": workspace_delta,
    "ruler_control_seed_spread": ruler,
    "bars": {
        "I1": I1_ALLOWANCE, "I2": I2_MARGIN, "I3": I3_MARGIN,
        "I4": I4_ALLOWANCE, "I5_entropy": I5_ENTROPY,
        "I5_rank": I5_RANK, "I5_probe": I5_PROBE,
    },
    "I1": bool(i1),
    "I2": bool(i2),
    "I3": bool(i3),
    "I4": bool(i4),
    "I5": bool(i5),
    "I6": bool(i6),
    "I7": bool(i7),
    "advance": advance,
    "decision": decision,
}
write_json("/kaggle/working/x40_summary.json", summary)
print("\nX40 VERDICT")
for key in ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "advance"):
    print(f"  {key}: {summary[key]}")
print(f"  decision: {decision}")
print(f"  val deltas (IP - control): {val_delta}")
print(f"  seen deltas (IP - control): {seen_delta}")
print(f"  unseen deltas (IP - pairs): {unseen_delta}")
print(f"  unseen deltas (pairs - control): {pairs_unseen_delta}")
print(f"  unseen deltas (workspace - pairs): {workspace_delta}")
print(f"  retention deltas (IP - control): {ret_delta}")
print(f"  control seed spread: {ruler}")
print(f"  workspace: {workspace}")
print("  summary written to /kaggle/working/x40_summary.json")
