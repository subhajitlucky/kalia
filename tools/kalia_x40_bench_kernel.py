"""Kaggle CPU kernel: score the archived X40 checkpoints on the 2x2 benchmark.

This is a diagnostic, not a re-score of X40's registered bars. It attaches the
completed `kalia-x40-ip` kernel output (the checkpoints) and the
`kalia-code-dev` snapshot, regenerates the deterministic benchmark, and scores
`control`, `pairs` and `IP` at both seeds. The question is narrow:

    was X40's composition failure real, or was it the unfair template?

Outputs: `x40_bench_summary.json`, `x40_bench_rows.jsonl`, `bench/audit.json`.
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

KERNEL_ID = "subhajitlucky/kalia-x40-bench"
KERNEL_REVISION = "unknown"
N_PER_CELL = 400
MODES = ("instruction_prefix", "edited_prefix")
MODELS = (
    "x40-control-s1338", "x40-control-s1339",
    "x40-pairs-s1338", "x40-pairs-s1339",
    "x40-ip-s1338", "x40-ip-s1339",
)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- code -------------------------------------------------------------------
hits = sorted(glob.glob("/kaggle/input/**/composition_benchmark.py", recursive=True))
preferred = [p for p in hits if "kalia-code-dev" in p]
assert preferred, (
    "kalia-code-dev not attached or lacks composition_benchmark.py; matches: "
    + str(hits)
)
code_root = os.path.dirname(preferred[0])
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in (
    "composition_benchmark.py", "eval_ip.py", "eval_reversibility.py",
    "intervention_pairs.py", "model.py",
):
    assert os.path.exists(req), f"{req} missing from the code snapshot"
print("code:", code_root)

# --- benchmark ---------------------------------------------------------------
print("\n===== X40 2x2 benchmark =====")
subprocess.run(
    [sys.executable, "composition_benchmark.py", "--out-dir",
     "/kaggle/working/bench", "--seed", "2026", "--n-per-cell", str(N_PER_CELL)],
    check=True,
)
bench_audit = json.loads(Path("/kaggle/working/bench/audit.json").read_text())
print("cells:", {k: v["n"] for k, v in bench_audit["cells"].items()})

# --- archived checkpoints ----------------------------------------------------
ckpts = sorted(glob.glob("/kaggle/input/**/x40-*/ckpt.pt", recursive=True))
preferred_ckpts = [p for p in ckpts if "kalia-x40-ip" in p] or ckpts
assert preferred_ckpts, "no X40 checkpoints found; attach kalia-x40-ip"
selected = [
    p for p in preferred_ckpts
    if Path(p).parent.name in MODELS
]
assert len(selected) == len(MODELS), (
    "expected the six control/pairs/IP checkpoints, found "
    + str([Path(p).parent.name for p in selected])
)
print("checkpoints:", [str(Path(p).parent.name) for p in selected])

# --- score -------------------------------------------------------------------
from eval_ip import accuracy_from_scores, score_candidates  # noqa: E402
from eval_reversibility import load_model  # noqa: E402
from intervention_pairs import InterventionDataset  # noqa: E402

device = torch.device("cpu")
datasets = {
    mode: InterventionDataset(
        Path("/kaggle/working/bench/bench.jsonl"),
        prefix_len=64, suffix_len=64, prompt_mode=mode,
    )
    for mode in MODES
}
summary = {
    "experiment": "x40-bench",
    "kernel_id": KERNEL_ID,
    "kernel_revision": KERNEL_REVISION,
    "n_per_cell": N_PER_CELL,
    "benchmark": bench_audit,
    "x40_source_kernel": "subhajitlucky/kalia-x40-ip",
    "models": {},
    "checkpoints": {},
}
rows_path = Path("/kaggle/working/x40_bench_rows.jsonl")
with open(rows_path, "w") as rows_fh:
    for ckpt in selected:
        name = Path(ckpt).parent.name
        print(f"\n=== {name} ===")
        model = load_model(ckpt, device)
        summary["models"][name] = {}
        for mode in MODES:
            ds = datasets[mode]
            scores = score_candidates(model, ds, device, batch_size=16)
            accuracy = accuracy_from_scores(ds, scores)
            summary["models"][name][mode] = {
                "by_cell": accuracy["by_cell"],
                "by_operation": accuracy["by_operation"],
                "singles": accuracy["seen"],
                "pairs": accuracy["unseen"],
                "overall": accuracy["overall"],
            }
            for row in accuracy["rows"]:
                rows_fh.write(json.dumps(
                    {"model": name, "prompt_mode": mode, **row},
                    sort_keys=True,
                ) + "\n")
            print(f"  prompt={mode}")
            for cell in ("S-S", "S-P", "U-S", "U-P"):
                b = accuracy["by_cell"].get(cell)
                if b:
                    print(
                        f"    {cell}: n={b['n']} acc={b['accuracy']:.4f} "
                        f"chance={b['chance']:.2f} above={b['accuracy']-b['chance']:+.4f}"
                    )
        summary["checkpoints"][name] = {
            "path": ckpt,
            "bytes": os.path.getsize(ckpt),
            "sha256": sha256_file(ckpt),
        }

# Attach the X40 verdict for context; the benchmark does not change it.
verdict_hits = sorted(glob.glob("/kaggle/input/**/x40_summary.json", recursive=True))
verdict_hits = [p for p in verdict_hits if "kalia-x40-ip" in p] or verdict_hits
if verdict_hits:
    x40 = json.loads(Path(verdict_hits[0]).read_text())
    summary["x40_verdict"] = {
        key: x40.get(key) for key in
        ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "advance", "decision")
    }

Path("/kaggle/working/x40_bench_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("\nwrote /kaggle/working/x40_bench_summary.json")
