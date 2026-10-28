"""Kaggle CPU kernel: the registered v0.3.0 final evaluation.

Registered as ``docs/preregistrations/2026-10-27-v030-second-epoch.md``. The
training kernel finished the 2,000-step re-warm update. This kernel attaches its
final checkpoint and runs the three registered numbers on CPU, at zero GPU
quota:

1. **E-1** — canonical 100-batch val (`eval_val.py`), bar ≤ 2.8048.
2. **E-2** — 2,000-sample benchmark vs v0.2.0, no task regressing by >2σ.
3. **E-3** — CL-0 probe on the final checkpoint, not worse than v0.2.0 by
   more than 0.0070.

Outputs: ``eval_val.md``, ``bench_v030_2k.json``, ``cl0_v030.jsonl``,
``v030_final.json``, ``run_manifest.json``.
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

ARM = "final"
KERNEL_ID = "unknown"  # staged manually; set below before push
KERNEL_REVISION = "unknown"  # replaced with the commit hash before push

V020_VAL = 2.8248
E1_BAR = 2.8048
V020_CL0 = 2.798875
E3_ALLOWANCE = 0.0070
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
hits = sorted(glob.glob("/kaggle/input/**/eval_bench.py", recursive=True))
preferred = [p for p in hits if "/kalia-code-dev/" in p]
assert preferred, (
    "kalia-code-dev not attached or lacks eval_bench.py; other matches "
    "(do NOT trust them, they are stale bundles): " + str(hits)
)
code_root = os.path.dirname(preferred[0])
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("eval_bench.py", "eval_val.py", "kalia_lm.py", "model.py", "data.py",
            "eval_reversibility.py", "lambada_loader.py",
            "tools/forgetting_probe.py", "eval/ttt_probe_stories.json"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"
print("code:", code_root)

ckpt_hits = sorted(glob.glob("/kaggle/input/**/out/ckpt.pt", recursive=True))
preferred_ckpt = [p for p in ckpt_hits if "/kalia-v030-epoch2/" in p]
ckpt_hits = preferred_ckpt or ckpt_hits
assert ckpt_hits, "attach subhajitlucky/kalia-v030-epoch2"
CKPT = ckpt_hits[0]

val_hits = sorted(glob.glob("/kaggle/input/**/val.bin", recursive=True))
preferred_val = [p for p in val_hits if "/kalia-prep-v2b/" in p]
val_hits = preferred_val or val_hits
assert val_hits, "attach subhajitlucky/kalia-prep-v2b"
VAL = val_hits[0]

base_hits = sorted(glob.glob("/kaggle/input/**/bench_v020_2k.json", recursive=True))
assert base_hits, "attach subhajitlucky/kalia-bench-pair-2k for the v0.2.0 2k baseline"
BASE = json.loads(open(base_hits[0]).read())
print("checkpoint:", CKPT)
print("val:", VAL)
print("v0.2.0 2k baseline:", base_hits[0])

from huggingface_hub import hf_hub_download  # noqa: E402

probe = hf_hub_download("kalia-lm/kalia-v020", "corpus/probe_val.bin", repo_type="model")
probe_hash = sha256_file(probe)
assert probe_hash.startswith(PROBE_SHA), (
    f"probe shard hash {probe_hash[:12]} does not match the G2 pin {PROBE_SHA}"
)
print("probe:", probe, probe_hash[:12])

# --- install the benchmark harness ------------------------------------------
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "lm-eval==0.4.9",
     "transformers<5", "tiktoken", "pyyaml", "datasets"],
    check=True,
)

# --- E-1: canonical 100-batch val -------------------------------------------
val_md = "/kaggle/working/eval_val.md"
subprocess.run(
    [sys.executable, "eval_val.py", "--ckpt", CKPT, "--val-bin", VAL,
     "--batches", "100", "--batch-size", "8", "--seed", "1234",
     "--out", val_md, "--device", "cpu"],
    check=True,
)
md = Path(val_md).read_text()
val_loss = float(re.search(r"Val loss:\s*([0-9.]+)", md).group(1))
val_bpb = float(re.search(r"bpB:\s*([0-9.]+)", md).group(1))
print(f"E-1 val {val_loss:.4f} | bpB {val_bpb:.4f} | bar {E1_BAR:.4f}")
e1 = val_loss <= E1_BAR

# --- E-2: 2,000-sample benchmark --------------------------------------------
bench_out = "/kaggle/working/bench_v030_2k.json"
subprocess.run(
    [sys.executable, "eval_bench.py", "--ckpt", CKPT, "--limit", str(LIMIT),
     "--out", bench_out],
    check=True,
)
bench = json.loads(Path(bench_out).read_text())
rows = {}
for task in TASKS:
    if task not in bench or task not in BASE:
        continue
    v030 = float(bench[task].get(METRIC[task], float("nan")))
    v020 = float(BASE[task].get(METRIC[task], float("nan")))
    se030 = float(bench[task].get(STDERR[task], 0.0))
    se020 = float(BASE[task].get(STDERR[task], 0.0))
    z = (v030 - v020) / math.sqrt(se030 ** 2 + se020 ** 2) if (se030 or se020) else float("nan")
    rows[task] = {
        "v020": v020 * 100.0,
        "v030": v030 * 100.0,
        "delta_pp": (v030 - v020) * 100.0,
        "z": z,
        "regression_2sigma": bool(math.isfinite(z) and z < -2.0),
    }
e2 = bool(rows) and all(not r["regression_2sigma"] for r in rows.values())
print("E-2 benchmark:", json.dumps(rows, indent=2, sort_keys=True))

# --- E-3: CL-0 on the final checkpoint --------------------------------------
cl0_path = "/kaggle/working/cl0_v030.jsonl"
subprocess.run(
    [sys.executable, "tools/forgetting_probe.py", "--ckpt", CKPT,
     "--probe", probe, "--ledger", cl0_path, "--batches", "25",
     "--batch-size", "8", "--device", "cpu", "--commit", KERNEL_REVISION],
    check=True,
)
cl0_rows = [json.loads(line) for line in Path(cl0_path).read_text().splitlines() if line.strip()]
cl0 = cl0_rows[-1]
cl0_loss = float(cl0["probe_loss"])
cl0_drift = cl0_loss - V020_CL0
e3 = cl0_drift <= E3_ALLOWANCE
print(f"E-3 CL-0 {cl0_loss:.6f} vs {V020_CL0:.6f} | drift {cl0_drift:+.6f} | "
      f"allowance +{E3_ALLOWANCE:.4f}")

# --- verdict ----------------------------------------------------------------
summary = {
    "experiment": "v030-final",
    "checkpoint": CKPT,
    "val": {"loss": val_loss, "bpb": val_bpb, "v020": V020_VAL, "bar": E1_BAR},
    "E1": bool(e1),
    "benchmark": rows,
    "E2": bool(e2),
    "cl0": {"probe_loss": cl0_loss, "v020": V020_CL0, "drift": cl0_drift,
            "allowance": E3_ALLOWANCE},
    "E3": bool(e3),
    "release_candidate": bool(e1 and e2 and e3),
    "bars": {"E1": E1_BAR, "E2": "no task regresses by >2 sigma",
             "E3": E3_ALLOWANCE},
    "note": "the v0.2.0 2k baseline is the archived bench_v020_2k.json; the "
            "registered 500-sample verdict is not rewritten",
}
write_json("/kaggle/working/v030_final.json", summary)
print("\nv0.3.0 FINAL VERDICT")
for k in ("E1", "E2", "E3", "release_candidate"):
    print(f"  {k}: {summary[k]}")
print(f"  val {val_loss:.4f} (v0.2.0 {V020_VAL:.4f}, bar {E1_BAR:.4f})")
print(f"  CL-0 drift {cl0_drift:+.6f} (allowance +{E3_ALLOWANCE:.4f})")
print("verdict written to /kaggle/working/v030_final.json")

write_run_manifest(
    "/kaggle/working/run_manifest.json",
    experiment="v030-final",
    checkpoint={"path": CKPT, "sha256": sha256_file(CKPT)},
    val_bin={"path": VAL, "sha256": sha256_file(VAL)},
    baseline={"path": base_hits[0], "sha256": sha256_file(base_hits[0])},
    probe={"path": probe, "sha256": probe_hash},
    code_snapshot=code_root,
    code_fingerprint=code_fingerprint(code_root),
    bars={"E1": E1_BAR, "E3": E3_ALLOWANCE, "limit": LIMIT},
)
