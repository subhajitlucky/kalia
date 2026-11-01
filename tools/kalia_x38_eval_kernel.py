"""Kaggle GPU kernel: X38 evaluation only.

The training kernel completed all eight arms, then died in the evaluation loop
because it looked for ``model.up`` instead of ``model.crb.up``. The checkpoints
are intact. This kernel attaches that run's output and executes only the
registered evaluation:

- canonical 100-batch val
- the X37 swap-sensitivity probe
- the held-out paraphrase-distance probe
- greedy entity retention (secondary)
- stable-stream usage and relative change
- C-1..C-6 and the advance flag

No retraining. The failed run's manifest, configs and checkpoints are hashed and
carried into the eval manifest.
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
from datetime import datetime, timezone
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ARM = "crb-eval"
KERNEL_ID = "unknown"  # staging pins this
KERNEL_REVISION = "unknown"  # staging pins this

SEEDS = ("1338", "1339")
ARMS = ("control", "split", "loss", "crb")
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


# --- code and inputs ---------------------------------------------------------
hits = sorted(glob.glob("/kaggle/input/**/train.py", recursive=True))
preferred = [p for p in hits if "/kalia-code-dev/" in p]
assert preferred, "kalia-code-dev not attached or lacks train.py: " + str(hits)
code_root = os.path.dirname(preferred[0])
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in (
    "model.py", "crb_pairs.py", "eval_val.py", "eval_entities.py",
    "eval_reversibility.py", "eval/ttt_probe_stories.json",
):
    assert os.path.exists(req), f"{req} missing from the code snapshot"
print("code:", code_root)

# The failed run's output is the kernel source. Prefer its paths, then the
# checkpoint dataset that carries the stripped checkpoints and configs.
def pick(
    pattern: str,
    markers: tuple[str, ...] = ("/kalia-x38-crb/", "/kalia-x38-ckpts/"),
) -> list[str]:
    found = sorted(glob.glob(pattern, recursive=True))
    preferred = [p for p in found if any(m in p for m in markers)]
    return preferred or found


ckpts = pick("/kaggle/input/**/out/x38-*/ckpt.pt")
assert len(ckpts) == 8, f"expected 8 checkpoints, found {len(ckpts)}: {ckpts}"
ckpt_by_name = {Path(p).parent.name: p for p in ckpts}
for name in (f"x38-{arm}-s{seed}" for seed in SEEDS for arm in ARMS):
    assert name in ckpt_by_name, f"missing checkpoint for {name}"
print("checkpoints:", len(ckpts))

configs = pick("/kaggle/input/**/configs/x38-*.yaml")
assert len(configs) >= 8, f"expected 8 configs, found {len(configs)}"
config_by_name = {Path(p).stem: p for p in configs}
for name in ckpt_by_name:
    assert name in config_by_name, f"missing config for {name}"

manifests = pick("/kaggle/input/**/run_manifest.json")
assert manifests, "the failed run's run_manifest.json is not attached"
failed_manifest = json.loads(Path(manifests[0]).read_text())
assert failed_manifest.get("experiment") == "crb", failed_manifest.get("experiment")
assert len(failed_manifest["arms"]) == 8
print("failed run manifest:", manifests[0])

data_hits = sorted(glob.glob("/kaggle/input/**/val.bin", recursive=True))
preferred_data = [p for p in data_hits if "/kalia-prep-v2b/" in p]
data_hits = preferred_data or data_hits
assert data_hits, "attach subhajitlucky/kalia-prep-v2b for val.bin"
val_bin = data_hits[0]
print("val:", val_bin)

# --- deterministic pairs, checked against the failed run ---------------------
pairs_dir = Path("/kaggle/working/crb")
subprocess.run(
    [sys.executable, "crb_pairs.py", "--out-dir", str(pairs_dir),
     "--seed", SEEDS[0], "--n-train", "1200", "--n-eval", "200"],
    check=True,
)
assert sha256_file(pairs_dir / "eval.jsonl") == failed_manifest["pairs"]["eval_sha256"]
assert sha256_file(pairs_dir / "train.jsonl") == failed_manifest["pairs"]["train_sha256"]
assert sha256_file(pairs_dir / "audit.json") == failed_manifest["pairs"]["audit_sha256"]
pair_audit = json.loads((pairs_dir / "audit.json").read_text())
assert not set(pair_audit["train_templates"]) & set(pair_audit["eval_templates"])
assert not set(pair_audit["train_names"]) & set(pair_audit["eval_names"])
print("pairs: hashes match the failed run")

# --- evaluation --------------------------------------------------------------
import tiktoken  # noqa: E402
import yaml  # noqa: E402
from data import TokenDataset  # noqa: E402
from eval_entities import extract_entities, retention  # noqa: E402
from eval_reversibility import load_model  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = tiktoken.get_encoding("gpt2")
stories = json.loads(Path("eval/ttt_probe_stories.json").read_text())


def run_final_val(name: str) -> float:
    ckpt = ckpt_by_name[name]
    md = Path("/kaggle/working/eval") / f"{name}.md"
    subprocess.run(
        [sys.executable, "eval_val.py", "--ckpt", ckpt, "--val-bin", val_bin,
         "--batches", "100", "--batch-size", "8", "--seed", "1234",
         "--out", str(md), "--device", str(device)],
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
        ckpt = ckpt_by_name[name]
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
            assert model.crb is not None, f"{name} has no CRB module"
            stable[name] = model.stable_stats(x_val)
            cfg = yaml.safe_load(open(config_by_name[name]))
            torch.manual_seed(int(seed))
            initial = GPT(GPTConfig(**cfg["model"]))
            rel = {}
            for key in ("up", "down"):
                a = getattr(initial.crb, key).weight.detach()
                b = getattr(model.crb, key).weight.detach().cpu()
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
    "source_kernel": "subhajitlucky/kalia-x38-crb",
    "source_manifest_sha256": sha256_file(manifests[0]),
    "seeds": [int(s) for s in SEEDS],
    "vals": vals,
    "swap": swaps,
    "paraphrase": paras,
    "retention": rets,
    "stable": stable,
    "params": {name: b["params"] for name, b in failed_manifest["arms"].items()},
    "bars": {"C1": C1_ALLOWANCE, "C2": C2_FLOOR, "C3": C3_MARGIN,
             "C4": C4_ALLOWANCE, "C5_rank": C5_RANK, "C5_norm": C5_NORM,
             "C5_change": C5_CHANGE},
    "val_delta_crb_minus_split": val_delta,
    "val_guard_split_minus_control": val_guard,
    "swap_delta_crb_minus_split": swap_delta,
    "paraphrase_delta_split_minus_crb": para_delta,
    "retention_delta_crb_minus_split": ret_delta,
    "split_swap_spread": split_swap_spread,
    "C1": bool(c1), "C1b": bool(c1b), "C2": bool(c2), "C3": bool(c3),
    "C4": bool(c4), "C5": bool(c5), "C6": bool(c6), "advance": advance,
}
write_json("/kaggle/working/x38_summary.json", summary)

write_json(
    "/kaggle/working/x38_eval_manifest.json",
    {
        "kernel_id": KERNEL_ID,
        "kernel_revision": KERNEL_REVISION,
        "recorded": datetime.now(timezone.utc).isoformat(),
        "source_kernel": "subhajitlucky/kalia-x38-crb",
        "source_manifest": manifests[0],
        "source_manifest_sha256": sha256_file(manifests[0]),
        "checkpoints": {name: {"path": path, "sha256": sha256_file(path)}
                        for name, path in sorted(ckpt_by_name.items())},
        "configs": {name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in sorted(config_by_name.items())},
        "pairs": {
            "train_sha256": sha256_file(pairs_dir / "train.jsonl"),
            "eval_sha256": sha256_file(pairs_dir / "eval.jsonl"),
            "audit_sha256": sha256_file(pairs_dir / "audit.json"),
        },
        "val_bin": {"path": val_bin, "sha256": sha256_file(val_bin)},
        "code_snapshot": code_root,
        "code_fingerprint": code_fingerprint(code_root),
        "bars": summary["bars"],
    },
)
print("\nX38 EVAL VERDICT")
for key in ("C1", "C1b", "C2", "C3", "C4", "C5", "C6", "advance"):
    print(f"  {key}: {summary[key]}")
print(f"  val deltas (crb - split): {val_delta}")
print(f"  swap deltas (crb - split): {swap_delta} | threshold {c2_threshold:.5f}")
print(f"  paraphrase deltas (split - crb): {para_delta}")
print(f"  stable: {stable}")
print("  summary written to /kaggle/working/x38_summary.json")
