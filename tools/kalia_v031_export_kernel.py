"""Kaggle CPU kernel: export KALIA v0.3.1 and re-score the written safetensors.

The merged checkpoint `best_merged.pt` is a training checkpoint (optimizer
state, internal names, no loadable config). This kernel runs
`tools/export_hf_model.py --release v031`, which writes `model.safetensors`,
`config.json`, `generation_config.json`, `model.py`, `sample.py`, `LICENSE`,
then loads the **written tensors** back through the repo's own model and
re-scores the registered 100-batch protocol. It refuses to pass unless the
artifact returns 2.7579.

It also stages the new v0.3.1 card and the GPT-2 tokenizer files from the
v0.2.0 repo. Nothing is uploaded; the release directory is the artifact.

CPU only: no GPU quota.
"""

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

KERNEL_ID = "unknown"
KERNEL_REVISION = "unknown"
EXPECTED_VAL = 2.7579
EXPECTED_BPB = 0.9016


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
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
hits = pick("/kaggle/input/**/export_hf_model.py", ("/kalia-code-dev/",))
assert hits, "kalia-code-dev not attached or lacks export_hf_model.py"
root_hits = pick("/kaggle/input/**/model.py", ("/kalia-code-dev/",))
assert root_hits, "kalia-code-dev not attached or lacks model.py"
code_root = os.path.dirname(root_hits[0])
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("tools/export_hf_model.py", "model.py", "sample.py", "data.py",
            "eval_val.py", "hf-model-card-v031.md", "LICENSE"):
    assert os.path.exists(req), f"{req} missing from the code snapshot"
print("code:", code_root)

ckpt_hits = pick("/kaggle/input/**/best_merged.pt", ("/kalia-v031-materialize/",))
assert ckpt_hits, "attach subhajitlucky/kalia-v031-materialize"
CKPT = ckpt_hits[0]
val_hits = pick("/kaggle/input/**/val.bin", ("/kalia-prep-v2b/",))
assert val_hits, "attach subhajitlucky/kalia-prep-v2b for val.bin"
VAL = val_hits[0]
print("checkpoint:", CKPT)
print("val:", VAL)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "safetensors",
     "tiktoken<1", "pyyaml", "numpy", "pandas"],
    check=True,
)

out = Path("/kaggle/working/release")
cmd = [
    sys.executable, "tools/export_hf_model.py",
    "--ckpt", CKPT, "--out", str(out), "--release", "v031", "--val-bin", VAL,
]
print("$", " ".join(cmd), flush=True)
proc = subprocess.run(cmd, capture_output=True, text=True)
print(proc.stdout)
if proc.stderr:
    print(proc.stderr)
assert proc.returncode == 0, f"export gate failed with {proc.returncode}"

shutil.copyfile("hf-model-card-v031.md", out / "README.md")

# The tokenizer travels with the weights. The model uses GPT-2 BPE, so take the
# public `gpt2` tokenizer files; no dependence on the private v0.2.0 repo.
tokenizer_files = []
try:
    from huggingface_hub import hf_hub_download
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "vocab.json", "merges.txt"):
        try:
            src = hf_hub_download("gpt2", name)
            shutil.copyfile(src, out / name)
            tokenizer_files.append(name)
        except Exception as exc:  # a missing optional file must not hide the weights
            print(f"tokenizer {name}: {type(exc).__name__}: {exc}")
except Exception as exc:
    print("tokenizer download skipped:", type(exc).__name__, exc)
print("tokenizer files:", tokenizer_files)

for name in ("model.safetensors", "config.json", "generation_config.json",
             "model.py", "sample.py", "LICENSE", "README.md"):
    assert (out / name).exists(), f"{name} missing from the release"

m = re.search(
    r"verification on the written tensors: val ([0-9.]+) / bpB ([0-9.]+)",
    proc.stdout,
)
assert m, "could not parse the export verification"
written_val, written_bpb = float(m.group(1)), float(m.group(2))
assert abs(written_val - EXPECTED_VAL) <= 5e-4, (written_val, EXPECTED_VAL)
assert abs(written_bpb - EXPECTED_BPB) <= 5e-4, (written_bpb, EXPECTED_BPB)

manifest = {
    "kernel_id": KERNEL_ID,
    "kernel_revision": KERNEL_REVISION,
    "recorded": datetime.now(timezone.utc).isoformat(),
    "release": "v031",
    "weights": {"v020": 0.3, "v030": 0.7},
    "checkpoint": {"path": CKPT, "sha256": sha256_file(CKPT)},
    "val_bin": {"path": VAL, "sha256": sha256_file(VAL)},
    "written_safetensors": {
        "path": str(out / "model.safetensors"),
        "sha256": sha256_file(out / "model.safetensors"),
        "bytes": (out / "model.safetensors").stat().st_size,
    },
    "written_val": written_val,
    "written_bpb": written_bpb,
    "expected_val": EXPECTED_VAL,
    "expected_bpb": EXPECTED_BPB,
    "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
              for p in sorted(out.iterdir()) if p.is_file()},
    "tokenizer_files": tokenizer_files,
    "export_stdout": proc.stdout,
}
write_json("/kaggle/working/v031_export_manifest.json", manifest)
print("\nV0.3.1 EXPORT VERDICT")
print(f"  written safetensors: val {written_val} / bpB {written_bpb} "
      f"({manifest['written_safetensors']['bytes']:,} bytes)")
print(f"  expected: {EXPECTED_VAL} / {EXPECTED_BPB}")
print("  PASS: the written artifact reproduces the registered number")
print("  release dir:", sorted(p.name for p in out.iterdir()))
