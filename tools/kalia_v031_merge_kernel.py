"""Kaggle CPU kernel: v0.3.1 — can a v0.2.0/v0.3.0 weight merge recover the
final-checkpoint regression?

The v0.3.0 second epoch's best in-training 100-batch val was 2.7255 at step
6,500; the final checkpoint at 6,770 scored 2.7743. The step-6,500 checkpoint
was overwritten (one `ckpt.pt` every 30 minutes, no EMA). This kernel tests
whether a weighted average of the v0.2.0 and v0.3.0 weights recovers some of the
gap, at zero GPU cost.

Protocol:
- v0.2.0 weight 0.0 / 0.1 / 0.2 / 0.3; v0.3.0 weight 1-w.
- w=0.0 must reproduce v0.3.0's 2.7743.
- Best merge -> 2k benchmark + CL-0.
- Release only if val < 2.7743, no 2k task regresses by >2σ vs v0.3.0, and
  CL-0 ≤ 2.763648 (v0.3.0's 2.756648 + 0.0070).

CPU only: no GPU quota is used.
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

KERNEL_ID = "unknown"  # staging pins this
KERNEL_REVISION = "unknown"  # staging pins this

WEIGHTS = (0.0, 0.1, 0.2, 0.3)  # weight on v0.2.0
V030_VAL = 2.7743
V020_VAL = 2.8248
V030_CL0 = 2.756648
CL0_ALLOWANCE = 0.0070
LIMIT = 2000
TASKS = ("arc_easy", "piqa", "hellaswag", "lambada_openai", "winogrande")
METRIC = {
    "arc_easy": "acc,none",
    "piqa": "acc,none",
    "hellaswag": "acc_norm,none",
    "lambada_openai": "acc,none",
    "winogrande": "acc,none",
}
STDERR = {
    "arc_easy": "acc_stderr,none",
    "piqa": "acc_stderr,none",
    "hellaswag": "acc_norm_stderr,none",
    "lambada_openai": "acc_stderr,none",
    "winogrande": "acc_stderr,none",
}
PROBE_SHA = "1664f71a"


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


def pick(pattern: str, markers: tuple[str, ...]) -> list[str]:
    found = sorted(glob.glob(pattern, recursive=True))
    preferred = [p for p in found if any(m in p for m in markers)]
    return preferred or found


# --- code and inputs ---------------------------------------------------------
hits = pick("/kaggle/input/**/eval_bench.py", ("/kalia-code-dev/",))
assert hits, "kalia-code-dev not attached or lacks eval_bench.py"
code_root = os.path.dirname(hits[0])
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in (
    "eval_bench.py", "eval_val.py", "kalia_lm.py", "model.py", "data.py",
    "eval_reversibility.py", "lambada_loader.py", "tools/forgetting_probe.py",
):
    assert os.path.exists(req), f"{req} missing from the code snapshot"
print("code:", code_root)

v020_hits = pick("/kaggle/input/**/kalia-v020-ckpt/ckpt.pt", ("/kalia-v020-ckpt/",))
assert v020_hits, "attach subhajitlucky/kalia-v020-ckpt"
V020_CKPT = v020_hits[0]

v030_hits = pick("/kaggle/input/**/out/ckpt.pt", ("/kalia-v030-epoch2/",))
assert v030_hits, "attach subhajitlucky/kalia-v030-epoch2"
V030_CKPT = v030_hits[0]

val_hits = pick("/kaggle/input/**/val.bin", ("/kalia-prep-v2b/",))
assert val_hits, "attach subhajitlucky/kalia-prep-v2b for val.bin"
VAL = val_hits[0]

base020_hits = pick("/kaggle/input/**/bench_v020_2k.json", ("/kalia-bench-pair-2k/",))
assert base020_hits, "attach subhajitlucky/kalia-bench-pair-2k for the v0.2.0 2k baseline"
BASE020 = json.loads(Path(base020_hits[0]).read_text())

v030_final_hits = pick("/kaggle/input/**/v030_final.json", ("/kalia-v030-final-eval/",))
assert v030_final_hits, "attach subhajitlucky/kalia-v030-final-eval"
V030_FINAL = json.loads(Path(v030_final_hits[0]).read_text())
v030_bench_hits = pick("/kaggle/input/**/bench_v030_2k.json", ("/kalia-v030-final-eval/",))
assert v030_bench_hits, "attach subhajitlucky/kalia-v030-final-eval for bench_v030_2k.json"
V030_BENCH = json.loads(Path(v030_bench_hits[0]).read_text())
cl0_hits = pick("/kaggle/input/**/cl0_v030.jsonl", ("/kalia-v030-final-eval/",))
assert cl0_hits, "attach subhajitlucky/kalia-v030-final-eval for cl0_v030.jsonl"
V030_CL0_ROW = json.loads(Path(cl0_hits[0]).read_text().strip().splitlines()[-1])

print("v0.2.0:", V020_CKPT)
print("v0.3.0:", V030_CKPT)
print("val:", VAL)
print("v0.3.0 baseline:", V030_VAL, "| CL-0:", V030_CL0)
assert abs(float(V030_FINAL["val"]["loss"]) - V030_VAL) < 1e-6
assert abs(float(V030_FINAL["cl0"]["probe_loss"]) - V030_CL0) < 1e-6

# --- merge -------------------------------------------------------------------
a = torch.load(V020_CKPT, map_location="cpu", weights_only=False)
b = torch.load(V030_CKPT, map_location="cpu", weights_only=False)
for key in ("n_layer", "n_head", "n_embd", "context_len", "vocab_size"):
    assert a["config"]["model"][key] == b["config"]["model"][key], (key, a["config"]["model"], b["config"]["model"])
for key, default in (("qk_norm", False), ("logit_softcap", 0.0), ("nope_interval", 0),
                     ("gated_residual", False), ("branch_norm", False)):
    assert a["config"]["model"].get(key, default) == b["config"]["model"].get(key, default), key
assert set(a["model"]) == set(b["model"]), "parameter sets differ"
for key in a["model"]:
    assert a["model"][key].shape == b["model"][key].shape, key
print("checkpoints compatible: same model config, keys and shapes")

merge_dir = Path("/kaggle/working/merge")
merge_dir.mkdir(parents=True, exist_ok=True)


def merge(weight_v020: float, out_path: Path) -> Path:
    state = {}
    for key in b["model"]:
        merged = (1.0 - weight_v020) * b["model"][key].float() + weight_v020 * a["model"][key].float()
        state[key] = merged.to(b["model"][key].dtype)
    ckpt = {
        "model": state,
        "optimizer": b["optimizer"],
        "step": b["step"],
        "tokens": b["tokens"],
        "config": b["config"],
        "merged_from": [V020_CKPT, V030_CKPT],
        "merge_weights": [weight_v020, 1.0 - weight_v020],
    }
    torch.save(ckpt, out_path)
    return out_path


def run_val(ckpt: Path, out_md: Path) -> float:
    subprocess.run(
        [sys.executable, "eval_val.py", "--ckpt", str(ckpt), "--val-bin", VAL,
         "--batches", "100", "--batch-size", "8", "--seed", "1234",
         "--out", str(out_md), "--device", "cpu"],
        check=True,
    )
    text = out_md.read_text()
    return float(re.search(r"Val loss:\s*([0-9.]+)", text).group(1))


rows = {}
for weight in WEIGHTS:
    name = f"w{weight:.1f}"
    ckpt = merge_dir / f"v031-{name}.pt"
    merge(weight, ckpt)
    val = run_val(ckpt, merge_dir / f"val-{name}.md")
    rows[name] = {
        "v020_weight": weight,
        "v030_weight": 1.0 - weight,
        "val": val,
        "ckpt": str(ckpt),
        "ckpt_sha256": sha256_file(ckpt),
    }
    print(f"{name}: v0.2.0 {weight:.1f} / v0.3.0 {1-weight:.1f} -> val {val:.4f}")

baseline = rows["w0.0"]["val"]
assert abs(baseline - V030_VAL) <= 0.001, (
    f"w0.0 should reproduce v0.3.0's {V030_VAL}; got {baseline}"
)
best_name = min(
    (name for name, row in rows.items() if row["v020_weight"] > 0 and row["val"] < V030_VAL),
    key=lambda name: rows[name]["val"],
    default=None,
)
best = rows[best_name] if best_name else None
print("best merge:", best_name, best["val"] if best else None)

# --- 2k benchmark and CL-0 for the winner ------------------------------------
bench_rows = {}
no_regression = False
cl0 = None
cl0_pass = False
if best is not None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "lm-eval==0.4.9",
         "transformers<5", "tiktoken", "pyyaml", "datasets"],
        check=True,
    )
    bench_out = Path("/kaggle/working/bench_v031_2k.json")
    subprocess.run(
        [sys.executable, "eval_bench.py", "--ckpt", best["ckpt"],
         "--limit", str(LIMIT), "--out", str(bench_out)],
        check=True,
    )
    bench = json.loads(bench_out.read_text())
    for task in TASKS:
        if task not in bench or task not in V030_BENCH:
            continue
        v031 = float(bench[task].get(METRIC[task], float("nan")))
        v030 = float(V030_BENCH[task].get(METRIC[task], float("nan")))
        v020 = float(BASE020[task].get(METRIC[task], float("nan")))
        se031 = float(bench[task].get(STDERR[task], 0.0))
        se030 = float(V030_BENCH[task].get(STDERR[task], 0.0))
        z = (v031 - v030) / math.sqrt(se031 ** 2 + se030 ** 2) if (se031 or se030) else float("nan")
        bench_rows[task] = {
            "v020": v020 * 100.0,
            "v030": v030 * 100.0,
            "v031": v031 * 100.0,
            "delta_vs_v030_pp": (v031 - v030) * 100.0,
            "z_vs_v030": z,
            "regression_2sigma": bool(math.isfinite(z) and z < -2.0),
        }
    no_regression = bool(bench_rows) and all(not r["regression_2sigma"] for r in bench_rows.values())
    print("benchmark:", json.dumps(bench_rows, indent=2, sort_keys=True))

    from huggingface_hub import hf_hub_download
    probe = hf_hub_download("kalia-lm/kalia-v020", "corpus/probe_val.bin", repo_type="model")
    probe_hash = sha256_file(probe)
    assert probe_hash.startswith(PROBE_SHA), probe_hash
    cl0_path = Path("/kaggle/working/cl0_v031.jsonl")
    subprocess.run(
        [sys.executable, "tools/forgetting_probe.py", "--ckpt", best["ckpt"],
         "--probe", probe, "--ledger", str(cl0_path), "--batches", "25",
         "--batch-size", "8", "--device", "cpu", "--commit", KERNEL_REVISION],
        check=True,
    )
    cl0_row = json.loads(cl0_path.read_text().strip().splitlines()[-1])
    cl0 = float(cl0_row["probe_loss"])
    cl0_pass = (cl0 - V030_CL0) <= CL0_ALLOWANCE
    shutil.copyfile(best["ckpt"], "/kaggle/working/best_merged.pt")
    rows[best_name]["ckpt"] = "/kaggle/working/best_merged.pt"
    for name, row in rows.items():
        path = Path(row["ckpt"])
        if name != best_name and path.exists():
            path.unlink()
    Path(best["ckpt"]).unlink(missing_ok=True)
    print(f"CL-0 {cl0:.6f} vs v0.3.0 {V030_CL0:.6f} | drift {cl0 - V030_CL0:+.6f}")

release_candidate = bool(best is not None and no_regression and cl0_pass)
summary = {
    "experiment": "v031-merge",
    "date": "2026-11-01",
    "baseline_v030_val": V030_VAL,
    "baseline_v020_val": V020_VAL,
    "baseline_v030_cl0": V030_CL0,
    "cl0_allowance": CL0_ALLOWANCE,
    "weights_v020": list(WEIGHTS),
    "rows": rows,
    "best": best_name,
    "best_val": best["val"] if best else None,
    "win_on_val": bool(best is not None),
    "benchmark": bench_rows,
    "E2_no_regression": bool(no_regression),
    "cl0": cl0,
    "E3_cl0": bool(cl0_pass),
    "release_candidate": release_candidate,
    "note": (
        "no merge beat v0.3.0's 2.7743; keep v0.3.0 and move to the "
        "best-checkpoint continuation"
        if best is None
        else "the merge wins on val; the 2k and CL-0 gates decide the release"
    ),
}
write_json("/kaggle/working/v031_merge_summary.json", summary)

write_json(
    "/kaggle/working/v031_merge_manifest.json",
    {
        "kernel_id": KERNEL_ID,
        "kernel_revision": KERNEL_REVISION,
        "recorded": datetime.now(timezone.utc).isoformat(),
        "v020_checkpoint": {"path": V020_CKPT, "sha256": sha256_file(V020_CKPT)},
        "v030_checkpoint": {"path": V030_CKPT, "sha256": sha256_file(V030_CKPT)},
        "val_bin": {"path": VAL, "sha256": sha256_file(VAL)},
        "v030_final": {"path": v030_final_hits[0], "sha256": sha256_file(v030_final_hits[0])},
        "v030_bench": {"path": v030_bench_hits[0], "sha256": sha256_file(v030_bench_hits[0])},
        "v020_bench": {"path": base020_hits[0], "sha256": sha256_file(base020_hits[0])},
        "bars": {"val": V030_VAL, "cl0": V030_CL0 + CL0_ALLOWANCE, "limit": LIMIT},
        "code_snapshot": code_root,
        "code_fingerprint": code_fingerprint(code_root),
    },
)

print("\nV0.3.1 MERGE VERDICT")
for key in ("win_on_val", "E2_no_regression", "E3_cl0", "release_candidate"):
    print(f"  {key}: {summary[key]}")
print(f"  best: {best_name} | val {summary['best_val']} vs {V030_VAL}")
print("  summary written to /kaggle/working/v031_merge_summary.json")
