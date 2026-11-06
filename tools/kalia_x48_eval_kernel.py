"""Kaggle CPU kernel: evaluate one X48 arm on the 9-task suite.

Set ARM and TOKENIZER at the top. For the K-Tok arm, attach the K-Tok kernel
output and pass its tokenizer JSON. CPU only; no GPU quota.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ARM = "gpt2_1338"
TOKENIZER = None  # or a ktok_32k.json path fragment
TASKS = "arc_easy,arc_challenge,boolq,hellaswag,lambada_openai,openbookqa,piqa,sciq,winogrande"
LIMIT = 2000

hits = sorted(Path("/kaggle/input").rglob("eval_bench.py"))
preferred = [p for p in hits if "kalia-code-dev" in str(p)]
assert preferred, f"kalia-code-dev not attached: {hits}"
code_root = preferred[0].parent
work = Path("/kaggle/working/kalia")
if not work.exists():
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, str(work))

ckpt_hits = sorted(Path("/kaggle/input").rglob("ckpt.pt"))
preferred_ckpt = [p for p in ckpt_hits if ARM in str(p)]
assert preferred_ckpt, f"no ckpt for {ARM}: {ckpt_hits}"
ckpt = preferred_ckpt[0]
print("checkpoint:", ckpt, flush=True)

tok_arg = []
if TOKENIZER:
    tok_hits = sorted(Path("/kaggle/input").rglob(TOKENIZER))
    assert tok_hits, f"tokenizer {TOKENIZER} not attached"
    tok_arg = ["--tokenizer", str(tok_hits[0])]
    print("tokenizer:", tok_hits[0], flush=True)

try:
    import lm_eval  # noqa: F401
except Exception:
    print("installing lm-eval", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lm-eval"],
                   check=True)
try:
    import tiktoken  # noqa: F401
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"],
                   check=True)
if TOKENIZER:
    try:
        import tokenizers  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"],
                       check=True)

out = Path(f"/kaggle/working/x48_eval/{ARM}.json")
out.parent.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, "eval_bench.py", "--ckpt", str(ckpt),
       "--tasks", TASKS, "--limit", str(LIMIT), "--out", str(out),
       "--device", "cpu"] + tok_arg
print("cmd:", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
results = json.loads(out.read_text())
print(json.dumps(results, indent=2, sort_keys=True))
