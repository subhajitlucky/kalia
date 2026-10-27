"""Kaggle GPU kernel: X36 — does a self-supervised entity memory improve retention?

Registered as ``docs/preregistrations/2026-10-27-X36-entity-memory-micro.md``.
Eight 30M arms: base, parameter-matched control, copy-only, and the full
workspace + copy + recurrence model, at seeds 1338/1339, 763 steps.

The kernel trains the arms, then measures the recurrence-head AUC and the copy
accuracy on a held-out val batch. The entity retention probe is a separate
kernel (``kalia-x36-entity``) that attaches this kernel's checkpoints; the
pre-registered verdict is computed there.

Outputs: ``out/x36-<arm>-s<seed>/`` per arm (ckpt.pt, val_log.csv,
train_log.csv), the eight configs, ``x36_train.json`` and
``run_manifest.json``.
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ARM = "memory-micro"  # staging pins this
KERNEL_ID = "unknown"  # staging pins, from the committed metadata
KERNEL_REVISION = "unknown"  # staging pins, from git HEAD

SEEDS = ("1338", "1339")
ARMS = ("base", "control", "copy", "memory")
STEPS = 763
# The code snapshot predates committed micro-qk-s1338.yaml; the X31 kernel
# builds the QK arm from micro-muon.yaml the same way. Same 6x384x6 model.
BASE_CONFIG = "configs/micro-muon.yaml"

# Model flags per arm. Nothing else moves.
FLAGS = {
    "base": {},
    "control": {"memory_control": True, "memory_control_hidden": 1024},
    "copy": {"memory_copy": True},
    "memory": {
        "memory_slots": 32,
        "memory_chunk": 64,
        "memory_copy": True,
        "recurrence_weight": 0.1,
    },
}
EXPECTED_PARAMS = {
    "base": 29921280,
    "control": 30707712,
    "copy": 30216577,
    "memory": 30697378,
}
M1_FLOOR = 0.005
M2_ALLOWANCE = 0.010
M3_AUC = 0.60


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


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().cpu().flatten()
    labels = labels.detach().cpu().flatten().bool()
    n_pos = int(labels.sum())
    n_neg = int(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


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
for req in ("train.py", "ablate.py", "data.py", "model.py", "configs/" + BASE_CONFIG,
            "eval/ttt_probe_stories.json"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"
print("code:", code_root, "| arm:", ARM)

data_hits = sorted(glob.glob("/kaggle/input/**/train.bin", recursive=True))
preferred_data = [p for p in data_hits if "/kalia-prep-v2b/" in p]
data_hits = preferred_data or data_hits
assert data_hits, "attach subhajitlucky/kalia-prep-v2b for train.bin"
data_dir = os.path.dirname(data_hits[0])
assert os.path.exists(os.path.join(data_dir, "val.bin")), "val.bin missing"
print("data:", data_dir)

# --- build the eight configs and check the parameter match ------------------
import yaml  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

base = yaml.safe_load(open(BASE_CONFIG))
base["model"]["qk_norm"] = True
base["model"]["logit_softcap"] = 0.0
assert base["model"]["qk_norm"] is True
assert base["train"]["max_steps"] == STEPS
assert base["train"]["eval_interval"] == 100

built = {}
for seed in SEEDS:
    for arm in ARMS:
        cfg = deepcopy(base)
        cfg["train"]["seed"] = int(seed)
        cfg["model"].update(FLAGS[arm])
        name = f"x36-{arm}-s{seed}"
        path = Path("configs") / f"{name}.yaml"
        yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
        model = GPT(GPTConfig(**cfg["model"]))
        params = model.num_params()
        assert params == EXPECTED_PARAMS[arm], (arm, params, EXPECTED_PARAMS[arm])
        built[name] = {
            "arm": arm,
            "seed": int(seed),
            "params": params,
            "flags": FLAGS[arm],
            "config_sha256": sha256_file(path),
        }
        print(f"built {name}: {params:,} params, flags {FLAGS[arm]}")

control_params = built["x36-control-s1338"]["params"]
memory_params = built["x36-memory-s1338"]["params"]
assert abs(control_params - memory_params) / control_params < 0.001, (
    control_params, memory_params
)
print("parameter match: control", control_params, "memory", memory_params)

write_run_manifest(
    "/kaggle/working/run_manifest.json",
    experiment=ARM,
    seeds=[int(s) for s in SEEDS],
    steps=STEPS,
    base_config=BASE_CONFIG,
    base_config_sha256=sha256_file(BASE_CONFIG),
    arms=built,
    bars={"M1_floor": M1_FLOOR, "M2_allowance": M2_ALLOWANCE, "M3_auc": M3_AUC},
    code_snapshot=code_root,
    code_fingerprint=code_fingerprint(code_root),
    data_dir=data_dir,
    train_bin={"path": data_hits[0], "bytes": os.path.getsize(data_hits[0]),
               "sha256": sha256_file(data_hits[0])},
    val_bin={"path": os.path.join(data_dir, "val.bin"),
             "bytes": os.path.getsize(os.path.join(data_dir, "val.bin")),
             "sha256": sha256_file(os.path.join(data_dir, "val.bin"))},
)

# --- train the eight arms, paired by seed ------------------------------------
order = ",".join(f"x36-{arm}-s{seed}" for seed in SEEDS for arm in ARMS)
print(f"\n===== X36 ({ARM}) — arms in order: {order} =====")
proc = subprocess.run(
    [sys.executable, "ablate.py", "--arms", order, "--steps", str(STEPS),
     "--data-dir", data_dir, "--out-root", "/kaggle/working/out"],
)
if proc.returncode != 0:
    sys.exit(proc.returncode)


# --- val, recurrence AUC and copy accuracy on a held-out batch ---------------
def run_final_val(run: str) -> float:
    path = Path("/kaggle/working/out") / run / "val_log.csv"
    rows = path.read_text().strip().splitlines()[1:]
    return float(rows[-1].split(",")[1])


from data import TokenDataset  # noqa: E402
from eval_reversibility import load_model  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_ds = TokenDataset(os.path.join(data_dir, "val.bin"), base["model"]["context_len"])
gen = torch.Generator().manual_seed(1234)
x_val, y_val = val_ds.get_batch(16, device, gen)

aucs = {}
copy_acc = {}
for seed in SEEDS:
    for arm in ("copy", "memory"):
        run = f"x36-{arm}-s{seed}"
        ckpt = Path("/kaggle/working/out") / run / "ckpt.pt"
        if not ckpt.exists():
            print(f"{run}: no checkpoint; skipping aux metrics")
            continue
        model = load_model(ckpt, device)
        model.eval()
        captured = []
        copy_calls = []
        rhook = (
            model.recurrence.register_forward_hook(
                lambda m, i, o: captured.append(o.detach())
            )
            if getattr(model, "recurrence", None) is not None
            else None
        )
        chook = (
            model.copy.register_forward_hook(
                lambda m, i, o: copy_calls.append(
                    (i[0].detach(), i[2].detach(), i[3].detach(), i[4])
                )
            )
            if getattr(model, "copy", None) is not None
            else None
        )
        with torch.no_grad():
            model(x_val, y_val)
        if rhook is not None:
            rhook.remove()
        if chook is not None:
            chook.remove()
        if captured:
            r_logits = torch.cat(captured, dim=1)
            labels, mask = model._recurrence_labels(x_val)
            if mask.any():
                aucs[run] = roc_auc(r_logits[mask], labels[mask])
                print(f"{run}: recurrence AUC {aucs[run]:.4f}")
        if copy_calls:
            hits = total = 0
            for h_q, h_k, idx_k, q_start in copy_calls:
                copy = model.copy.distribution(h_q, h_k, idx_k, q_start)
                tgt = y_val[:, q_start:q_start + h_q.size(1)]
                in_prefix = (idx_k.unsqueeze(1) == tgt.unsqueeze(-1)).any(dim=-1)
                correct = copy.argmax(dim=-1) == tgt
                hits += int((correct & in_prefix).sum())
                total += int(in_prefix.sum())
            copy_acc[run] = (hits / total) if total else None
            print(f"{run}: copy accuracy {copy_acc[run]} ({hits}/{total})")

summary = {
    "experiment": ARM,
    "seeds": [int(s) for s in SEEDS],
    "steps": STEPS,
    "vals": {
        f"x36-{arm}-s{seed}": run_final_val(f"x36-{arm}-s{seed}")
        for seed in SEEDS for arm in ARMS
    },
    "params": {f"x36-{arm}-s{seed}": built[f"x36-{arm}-s{seed}"]["params"]
               for seed in SEEDS for arm in ARMS},
    "recurrence_auc": aucs,
    "copy_accuracy": copy_acc,
    "bars": {"M1_floor": M1_FLOOR, "M2_allowance": M2_ALLOWANCE, "M3_auc": M3_AUC},
    "note": "entity retention is measured by kalia-x36-entity; this summary is training + aux",
}
write_json("/kaggle/working/x36_train.json", summary)
print(json.dumps(summary, indent=2, sort_keys=True))
