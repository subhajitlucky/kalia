"""Kaggle GPU kernel: X36 entity probe — does the memory keep names?

Registered as ``docs/preregistrations/2026-10-27-X36-entity-memory-micro.md``.
Attaches the eight checkpoints from ``kalia-x36-memory`` and runs the short-context
entity protocol:

- 20 frozen stories, prompt truncated to 384 tokens;
- one greedy draw plus 8 candidates x 128 tokens at temperature 0.8 / top-k 200;
- split each continuation at 64 tokens; retention is scored on the second half.

The absolute numbers are not comparable to the 58M X34 protocol; the paired
control/memory contrast is the measurement. This kernel also computes the
greedy distinct-2 (copy degeneration) and writes the pre-registered verdict.

Outputs: ``x36_entity.json``, ``x36_verdict.json``, ``run_manifest.json``.
"""

import glob
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ARM = "entity"  # manual kernel; the id is below
KERNEL_ID = "subhajitlucky/kalia-x36-entity"
KERNEL_REVISION = "manual-2026-10-27"  # replaced with the commit hash before push

PROMPT_TOKENS = 384
GEN_TOKENS = 128
HALF = 64
N = 8
TEMPERATURE = 0.8
TOP_K = 200
SEED_BASE = 7
SEEDS = ("1338", "1339")
ARMS = ("base", "control", "copy", "memory")
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


def distinct2(text: str) -> float:
    toks = text.split()
    if len(toks) < 2:
        return 0.0
    bigrams = list(zip(toks, toks[1:]))
    return len(set(bigrams)) / len(bigrams)


# --- code, stories, training summary ----------------------------------------
hits = sorted(glob.glob("/kaggle/input/**/model.py", recursive=True))
preferred = [p for p in hits if "/kalia-code-dev/" in p]
assert preferred, (
    "kalia-code-dev not attached or lacks model.py; other matches "
    "(do NOT trust them, they are stale bundles): " + str(hits)
)
code_root = os.path.dirname(preferred[0])
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("model.py", "eval_entities.py", "eval_reversibility.py",
            "eval/ttt_probe_stories.json"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"
print("code:", code_root)

summary_hits = sorted(glob.glob("/kaggle/input/**/x36_train.json", recursive=True))
assert summary_hits, "attach subhajitlucky/kalia-x36-memory as a kernel source"
train_summary = json.load(open(summary_hits[0]))
print("train summary:", summary_hits[0])

import tiktoken  # noqa: E402
from eval_entities import retention  # noqa: E402
from eval_reversibility import load_model  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = tiktoken.get_encoding("gpt2")
stories = json.loads(Path("eval/ttt_probe_stories.json").read_text())
print("device:", device, "| stories:", len(stories))

results = {}
for seed in SEEDS:
    for arm in ARMS:
        run = f"x36-{arm}-s{seed}"
        ckpt_hits = sorted(glob.glob(f"/kaggle/input/**/out/{run}/ckpt.pt",
                                     recursive=True))
        assert ckpt_hits, f"{run}: checkpoint not found"
        model = load_model(ckpt_hits[0], device)
        model.eval()
        rows = []
        for i, story in enumerate(stories):
            ids = enc.encode_ordinary(story["prompt"])[:PROMPT_TOKENS]
            prompt_text = enc.decode(ids)
            prompt = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                greedy = model.generate(
                    prompt, max_new_tokens=GEN_TOKENS,
                    temperature=1e-6, top_k=None,
                )[0, prompt.size(1):].cpu().tolist()
                gen = torch.Generator(device=device).manual_seed(SEED_BASE + i)
                first, second, cands = [], [], []
                for _ in range(N):
                    c = model.generate(
                        prompt, max_new_tokens=GEN_TOKENS,
                        temperature=TEMPERATURE, top_k=TOP_K, generator=gen,
                    )[0, prompt.size(1):].cpu().tolist()
                    cands.append(c)
                    first.append(retention(prompt_text, enc.decode(c[:HALF])))
                    second.append(retention(prompt_text, enc.decode(c[HALF:])))
            greedy_text = enc.decode(greedy[HALF:])
            half_idx = max(range(N), key=lambda j: first[j])
            rows.append({
                "id": story["id"],
                "greedy": retention(prompt_text, greedy_text),
                "half": second[half_idx],
                "oracle": max(second),
                "distinct2": distinct2(greedy_text),
            })
        mean = lambda key: sum(r[key] for r in rows) / len(rows)
        results[run] = {
            "arm": arm,
            "seed": int(seed),
            "greedy": mean("greedy"),
            "half": mean("half"),
            "oracle": mean("oracle"),
            "distinct2": mean("distinct2"),
            "rows": rows,
        }
        print(f"{run}: greedy {results[run]['greedy']:.4f} | "
              f"half {results[run]['half']:.4f} | "
              f"oracle {results[run]['oracle']:.4f} | "
              f"distinct2 {results[run]['distinct2']:.4f}")

# --- the pre-registered verdict ---------------------------------------------
def avg(arm, key, seeds=SEEDS):
    return sum(results[f"x36-{arm}-s{s}"][key] for s in seeds) / len(seeds)


control_g = {"1338": results["x36-control-s1338"]["greedy"],
             "1339": results["x36-control-s1339"]["greedy"]}
memory_g = {"1338": results["x36-memory-s1338"]["greedy"],
            "1339": results["x36-memory-s1339"]["greedy"]}
deltas = {s: memory_g[s] - control_g[s] for s in SEEDS}
control_spread = abs(control_g["1338"] - control_g["1339"])
m1_bar = max(control_spread, M1_FLOOR)
m1 = (sum(deltas.values()) / len(deltas) >= m1_bar) and all(d > 0 for d in deltas.values())
m1b = avg("memory", "oracle") >= avg("control", "oracle") - M1_FLOOR

vals = train_summary.get("vals", {})
memory_val = sum(vals.get(f"x36-memory-s{s}", 1e9) for s in SEEDS) / len(SEEDS)
control_val = sum(vals.get(f"x36-control-s{s}", 1e9) for s in SEEDS) / len(SEEDS)
m2 = (memory_val - control_val) <= M2_ALLOWANCE

aucs = train_summary.get("recurrence_auc", {})
memory_auc = [aucs[f"x36-memory-s{s}"] for s in SEEDS if f"x36-memory-s{s}" in aucs]
m3 = bool(memory_auc) and (sum(memory_auc) / len(memory_auc) > M3_AUC)

base_d2 = avg("base", "distinct2")
copy_d2 = avg("copy", "distinct2")
memory_d2 = avg("memory", "distinct2")
m4 = min(copy_d2, memory_d2) >= 0.8 * base_d2

verdict = {
    "M1": bool(m1),
    "M1b": bool(m1b),
    "M2": bool(m2),
    "M3": bool(m3),
    "M4": bool(m4),
    "advance": bool(m1 and m2 and m3 and m4),
    "bars": {"M1_floor": M1_FLOOR, "M2_allowance": M2_ALLOWANCE, "M3_auc": M3_AUC},
    "control_spread": control_spread,
    "deltas": deltas,
    "m1_bar": m1_bar,
    "vals": {"memory": memory_val, "control": control_val, "delta": memory_val - control_val},
    "recurrence_auc": memory_auc,
    "distinct2": {"base": base_d2, "copy": copy_d2, "memory": memory_d2},
    "results": results,
}
write_json("/kaggle/working/x36_entity.json", results)
write_json("/kaggle/working/x36_verdict.json", verdict)
print("\nX36 VERDICT")
for key in ("M1", "M1b", "M2", "M3", "M4", "advance"):
    print(f"  {key}: {verdict[key]}")
print(f"  control spread: {control_spread:+.4f} | m1 bar: {m1_bar:+.4f}")
print(f"  greedy deltas: {deltas}")
print(f"  val: memory {memory_val:.4f} control {control_val:.4f} delta {memory_val - control_val:+.4f}")
print(f"  recurrence AUC: {memory_auc}")
print(f"  distinct2: base {base_d2:.3f} copy {copy_d2:.3f} memory {memory_d2:.3f}")
print("  verdict written to /kaggle/working/x36_verdict.json")

write_run_manifest(
    "/kaggle/working/run_manifest.json",
    experiment="x36-entity",
    seeds=[int(s) for s in SEEDS],
    prompt_tokens=PROMPT_TOKENS,
    gen_tokens=GEN_TOKENS,
    half=HALF,
    samples=N,
    code_snapshot=code_root,
    code_fingerprint=code_fingerprint(code_root),
    train_summary={"path": summary_hits[0], "sha256": sha256_file(summary_hits[0])},
    stories_sha256=sha256_file("eval/ttt_probe_stories.json"),
)
