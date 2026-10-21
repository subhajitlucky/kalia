"""Kaggle CPU kernel: the 2,000-sample benchmark pair, v0.1.2 vs v0.2.0.

Every zero-shot number in the v0.2.0 release was measured at `limit=500`, where
D45 measured lm-eval's standard error at **1.99-2.24 pp** — wider than most of
the movements being reported. The 2026-10-17 review recommended 2,000 samples;
`eval_bench.py` changed its default on 2026-10-18, but no 2,000-sample run was
ever archived. This kernel runs the same five tasks on both checkpoints at
2,000 samples, on CPU, so the comparison is finally resolved at ~1 pp.

Why both checkpoints in one kernel: the published v0.1.2 numbers are at 500
samples too. Comparing a 2,000-sample v0.2.0 against a 500-sample v0.1.2 would
re-introduce exactly the error bar this run exists to remove. Both models are
scored at 2,000, in one session, with one code snapshot.

The 500-sample numbers are not rewritten. They stay in the record with their
~2 pp error; this run is a new, higher-resolution observation of the same
comparison.

Outputs: ``bench_v012_2k.json``, ``bench_v020_2k.json``,
``bench_pair_2k.json``, ``run_manifest.json``. CPU only; no GPU quota.
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ARM = "2k"  # staging pins this; it names the experiment and its outputs
KERNEL_ID = "unknown"  # staging pins, from the committed metadata
KERNEL_REVISION = "unknown"  # staging pins, from git HEAD

LIMIT = 2000
TASKS = ("arc_easy", "piqa", "hellaswag", "lambada_openai", "winogrande")
# The metric each task's published card number uses. HellaSwag is acc_norm;
# the other four are acc. Changing this mapping would change the claim.
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
CHECKPOINTS = (
    ("v012", "kalia-train-v012"),
    ("v020", "kalia-train-v020"),
)


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


# --- code and checkpoints ---------------------------------------------------
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
for req in ("eval_bench.py", "kalia_lm.py", "model.py", "data.py",
            "eval_reversibility.py"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"
print("code:", code_root)

found = {}
for label, slug in CHECKPOINTS:
    ckpts = sorted(glob.glob(
        f"/kaggle/input/notebooks/subhajitlucky/{slug}/**/ckpt.pt",
        recursive=True,
    ))
    assert len(ckpts) == 1, (
        f"expected exactly one ckpt.pt under {slug}, found {ckpts}; "
        "refusing to guess which checkpoint is the final model"
    )
    found[label] = ckpts[0]
    print(f"{label}: {ckpts[0]} ({os.path.getsize(ckpts[0])} bytes)")

# Fail before the expensive install/run if the two checkpoints are the same
# file or the same weights. A benchmark pair that silently scores one model
# twice is the exact class of wrong-instrument failure this project keeps
# finding.
same_path = found["v012"] == found["v020"]
same_hash = sha256_file(found["v012"]) == sha256_file(found["v020"])
assert not same_path and not same_hash, (
    "v0.1.2 and v0.2.0 checkpoints are identical; the comparison would be "
    "a model against itself"
)
print("checkpoint pair is distinct")

# --- the harness: same pins as the 500-sample bench -------------------------
# lm-eval 0.4.9 imports transformers.AutoModelForVision2Seq, removed in
# transformers v5, and Kaggle's image now ships v5. Pin the pair.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "lm-eval==0.4.9",
     "transformers<5", "tiktoken", "pyyaml", "datasets"],
    check=True,
)
import importlib.metadata as md  # noqa: E402
print("versions:", {p: md.version(p) for p in
                    ("lm_eval", "transformers", "torch", "datasets")})

manifest = {
    "experiment": ARM,
    "kernel_id": KERNEL_ID,
    "kernel_revision": KERNEL_REVISION,
    "recorded": datetime.now(timezone.utc).isoformat(),
    "limit": LIMIT,
    "tasks": list(TASKS),
    "metric": METRIC,
    "stderr": STDERR,
    "code_snapshot": code_root,
    "code_fingerprint": code_fingerprint(code_root),
    "checkpoints": {
        label: {"path": path, "bytes": os.path.getsize(path),
                "sha256": sha256_file(path)}
        for label, path in found.items()
    },
}
write_json("/kaggle/working/run_manifest.json", manifest)
print("run manifest:", json.dumps(manifest, sort_keys=True))

# --- run both, at the same limit, in one session ---------------------------
raw = {}
for label, _ in CHECKPOINTS:
    out = f"/kaggle/working/bench_{label}_2k.json"
    print(f"\n===== {label}: {LIMIT} samples per task =====", flush=True)
    subprocess.run(
        [sys.executable, "eval_bench.py",
         "--ckpt", found[label],
         "--out", out,
         "--limit", str(LIMIT),
         "--tasks", ",".join(TASKS)],
        check=True,
    )
    raw[label] = json.loads(Path(out).read_text())
    print(f"{label} results:", json.dumps(raw[label], sort_keys=True))

# --- the comparison, with the error bar attached ---------------------------
def metric_value(label, task, key):
    row = raw[label].get(task, {})
    if key not in row:
        raise KeyError(f"{label}/{task}: {key} missing from lm-eval results")
    return float(row[key])


comparison = {}
print("\n" + "=" * 78)
print(f"KALIA benchmark pair — {LIMIT} samples/task, 0-shot")
print(f"{'task':<16s} {'v0.1.2':>8s} {'v0.2.0':>8s} {'delta pp':>9s} "
      f"{'z':>6s}  S-A (<=1pp regression)")
for task in TASKS:
    a = metric_value("v012", task, METRIC[task])
    b = metric_value("v020", task, METRIC[task])
    se_a = metric_value("v012", task, STDERR[task])
    se_b = metric_value("v020", task, STDERR[task])
    delta_pp = (b - a) * 100.0
    se_pp = (se_a ** 2 + se_b ** 2) ** 0.5 * 100.0
    z = delta_pp / se_pp if se_pp else 0.0
    regression = delta_pp < -1.0
    comparison[task] = {
        "metric": METRIC[task],
        "v012": a, "v020": b,
        "v012_stderr": se_a, "v020_stderr": se_b,
        "delta_pp": delta_pp, "z": z,
        "regression_over_1pp": regression,
    }
    print(f"{task:<16s} {a*100:8.2f} {b*100:8.2f} {delta_pp:+9.2f} "
          f"{z:6.2f}  {'FAIL' if regression else 'ok'}")
    if task == "lambada_openai":
        pa = metric_value("v012", task, "perplexity,none")
        pb = metric_value("v020", task, "perplexity,none")
        print(f"{'  lambada ppl':<16s} {pa:8.2f} {pb:8.2f} "
              f"{pb - pa:+9.2f}")

# Same S-A rule as the v0.2.0 registration: no task may regress by more than
# 1.0 pp. This run does not move the bar; it measures the delta more precisely.
fails = [t for t, row in comparison.items() if row["regression_over_1pp"]]
significant = [t for t, row in comparison.items() if abs(row["z"]) >= 2.0]
verdict = {
    "limit": LIMIT,
    "comparison": comparison,
    "s_a_fails": fails,
    "two_sigma_movements": significant,
    "note": (
        "S-A uses the registered 1.0 pp allowance. z is descriptive: at 2,000 "
        "samples, |z| >= 2 is the point where a movement is resolvable at 2 sigma."
    ),
}
write_json("/kaggle/working/bench_pair_2k.json", verdict)
print("\nS-A regressions >1pp:", fails or "none")
print("Movements at |z| >= 2:", significant or "none")
print("bench_pair_2k.json written")
