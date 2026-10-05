"""X25 smoke: end-to-end TTT measurement on CPU, 1 story, short continuation.

Validates the harness (checkpoint load -> generate with/without TTT ->
entity retention), not the hypothesis. Full 20-story measurement runs on GPU.
"""
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken
import torch

from eval_entities import entity_report
from model import GPT, GPTConfig
from sample import load_model
from ttt import generate_with_ttt

DEVICE = torch.device("cpu")
ROOT = Path(__file__).resolve().parent.parent
STORY = json.loads((ROOT / "eval" / "ttt_probe_stories.json").read_text())[0]
enc = tiktoken.get_encoding("gpt2")

print(f"story: {STORY['id']} {STORY['source'][:60]}", flush=True)
prompt_ids = torch.tensor([enc.encode(STORY["prompt"])], dtype=torch.long)
print(f"prompt tokens: {prompt_ids.size(1)}", flush=True)

model = load_model("/tmp/kalia-ckpt/checkpoints/ckpt.pt", DEVICE)
print("checkpoint loaded", flush=True)

results = {}
for tag, lr in (("control-lr0", 0.0), ("ttt", 1e-4)):
    start = time.time()
    out, diag = generate_with_ttt(
        model, prompt_ids, max_new_tokens=256, chunk_tokens=256,
        ttt_lr=lr, temperature=0.8, top_k=200, seed=7,
    )
    continuation = enc.decode(out[0, prompt_ids.size(1):].tolist())
    report = entity_report(STORY["prompt"], continuation)
    results[tag] = {
        "retention": report["retention"],
        "updates": diag["updates"],
        "seconds": round(time.time() - start, 1),
    }
    print(f"{tag}: retention={report['retention']} updates={diag['updates']} "
          f"({results[tag]['seconds']}s)", flush=True)

json.dump(results, open("/tmp/ttt_smoke.json", "w"), indent=2)
print("SMOKE DONE", flush=True)
