"""Kaggle CPU kernel: X41 end-to-end eval from saved compiler prefixes.

The GPU training job produced the LoRA adapters and generated prefixes; it
crashed only when a 65-token prefix hit the reader's 64-token window. This
kernel re-scores the saved generated prefixes with the fixed fallback guard.
No training and no generation, so it runs on CPU and spends no GPU quota.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_REPO = "kalia-lm/kalia-v031"
BASE_SHA = "bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468"
BENCH_SHA = "aad1b32187a8697b896f6be8408b979abd389466bd9fd42256b2f547e87dd45b"

# From the X41 training log; the compiler's singles-only dev exact match.
SINGLES_DEV_EM = {"B_1401": 0.9966, "B_1402": 0.9920, "A_1401": None}

# --- code -------------------------------------------------------------------
hits = sorted(Path("/kaggle/input").rglob("edit_compiler.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached or lacks edit_compiler.py: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))

# --- artifacts --------------------------------------------------------------
gen_hits = sorted(Path("/kaggle/input").rglob("generated_B_1401.jsonl"))
assert gen_hits, "kalia-x41-artifacts not attached"
art_root = gen_hits[0].parent
bench_path = art_root / "bench.jsonl"
assert bench_path.exists(), f"bench.jsonl missing from {art_root}"
print("code:", code_root)
print("artifacts:", art_root)

try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "safetensors", "huggingface_hub"], check=True)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

import torch  # noqa: E402

from model import GPT, GPTConfig  # noqa: E402
from edit_compiler import score_generated  # noqa: E402

torch.set_num_threads(min(8, os.cpu_count() or 1))
device = torch.device("cpu")

base_dir = Path("/kaggle/working/base")
base_dir.mkdir(parents=True, exist_ok=True)
sf = hf_hub_download(BASE_REPO, "model.safetensors", local_dir=str(base_dir))
cf = hf_hub_download(BASE_REPO, "config.json", local_dir=str(base_dir))
base_sha = hashlib.sha256(Path(sf).read_bytes()).hexdigest()
assert base_sha == BASE_SHA, (base_sha, BASE_SHA)
raw_cfg = json.loads(Path(cf).read_text())
cfg = {k: v for k, v in raw_cfg.items() if k in GPTConfig.__dataclass_fields__}
base = GPT(GPTConfig(**cfg))
load = base.load_state_dict(load_file(sf), strict=False)
assert not load.missing_keys and not load.unexpected_keys
base.to(device).eval()
for p in base.parameters():
    p.requires_grad = False

bench_sha = hashlib.sha256(bench_path.read_bytes()).hexdigest()
assert bench_sha == BENCH_SHA, (bench_sha, BENCH_SHA)
records = [json.loads(line) for line in bench_path.read_text().splitlines() if line.strip()]
by_id = {r["id"]: r for r in records}
print("base sha:", base_sha)
print("bench sha:", bench_sha, "records:", len(records))

try:
    import tiktoken
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"],
                   check=True)
    import tiktoken

enc = tiktoken.get_encoding("gpt2")
out_dir = Path("/kaggle/working/x41_eval")
out_dir.mkdir(parents=True, exist_ok=True)

summary = {
    "experiment": "X41-end2end-eval",
    "base_sha256": base_sha,
    "bench_sha256": bench_sha,
    "singles_dev_em_from_training_log": SINGLES_DEV_EM,
    "arms": {},
    "baseline": {},
}
for tag in ("normal", "gold"):
    p = art_root / f"baseline_{tag}.json"
    if p.exists():
        summary["baseline"][tag] = json.loads(p.read_text())

for tag in ("B_1401", "B_1402", "A_1401"):
    path = art_root / f"generated_{tag}.jsonl"
    assert path.exists(), f"missing {path}"
    gen_records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(gen_records) == len(records), (tag, len(gen_records), len(records))
    gen_by_id = {r["id"]: r["edited_prefix"] for r in gen_records}
    generated = [gen_by_id[r["id"]] for r in records]
    payload = score_generated(
        base, records, generated,
        out_dir / f"generated_{tag}_rescored.jsonl",
        out_dir / f"end2end_{tag}.json",
        out_dir / f"end2end_{tag}_rows.jsonl",
        device, batch_size=16, enc=enc, max_prefix_tokens=64,
    )
    summary["arms"][tag] = payload
    print(f"\n{tag}: compiler_em {payload['compiler']['exact_match']:.4f} "
          f"fallback {payload['compiler']['fallback_rate']:.4f}")
    print("  cells:", {k: round(v["accuracy"], 4)
                       for k, v in payload["by_cell"].items()})
    print("  ops:", {k: round(v["accuracy"], 4)
                     for k, v in sorted(payload["by_operation"].items())})


def cell(arm, name):
    return summary["arms"][arm]["by_cell"][name]["accuracy"]


def op(arm, name):
    return summary["arms"][arm]["by_operation"][name]["accuracy"]


b = ["B_1401", "B_1402"]
summary["bars"] = {
    "C-1_singles_em": all(SINGLES_DEV_EM[a] >= 0.90 for a in b),
    "C-2_number_place": all(op(a, "number+place") >= 0.40 for a in b),
    "C-3_rename_swap": all(op(a, "rename+swap") >= 0.45 for a in b),
    "C-4_SP": all(cell(a, "S-P") >= 0.40 for a in b),
    "C-5_UP": all(cell(a, "U-P") >= 0.30 for a in b),
    "C-6_base_frozen": base_sha == BASE_SHA and all(
        not p.requires_grad for p in base.parameters()
    ),
    "values": {
        a: {
            "singles_em": SINGLES_DEV_EM[a],
            "number+place": op(a, "number+place"),
            "rename+swap": op(a, "rename+swap"),
            "S-P": cell(a, "S-P"),
            "U-P": cell(a, "U-P"),
            "compiler_em": summary["arms"][a]["compiler"]["exact_match"],
            "fallback_rate": summary["arms"][a]["compiler"]["fallback_rate"],
        }
        for a in summary["arms"]
    },
}
(out_dir / "x41_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("\nX41 END-TO-END VERDICT")
for k, v in summary["bars"].items():
    if k != "values":
        print(f"  {k}: {v}")
def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


for tag, row in summary["bars"]["values"].items():
    print(f"  {tag}: " + " ".join(f"{k}={_fmt(v)}" for k, v in row.items()))
print("  summary: /kaggle/working/x41_eval/x41_summary.json")
