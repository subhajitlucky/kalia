"""Kaggle GPU kernel: entity-verified decoding measurement.

Three selectors over the same model and the same 8 samples per story: greedy
(1 draw), log-probability pick, and entity pick. Metric: entity retention and
mean span via `eval_entities.py`, on the frozen Gutenberg probe stories
(`eval/ttt_probe_stories.json`, the X25 measuring set). The reference
continuation is scored too, as the ceiling real prose reaches.

Measurement, not a claim; results go to `docs/results/`.
"""

import glob
import json
import os
import shutil
import subprocess
import sys

work = "/kaggle/working/kalia"
if not os.path.exists(work):
    hits = sorted(glob.glob("/kaggle/input/**/entity_decode.py", recursive=True))
    preferred = [p for p in hits if "/kalia-code-dev/" in p]
    assert preferred, (
        "kalia-code-dev not attached or lacks entity_decode.py; other matches "
        "(do NOT trust them, they are stale bundles): " + str(hits)
    )
    shutil.copytree(os.path.dirname(preferred[0]), work)
os.chdir(work)
sys.path.insert(0, work)
for req in (
    "entity_decode.py", "eval_entities.py", "best_of_n.py", "model.py",
    "sample.py", "eval/ttt_probe_stories.json",
):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"

try:
    import tiktoken  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"], check=True)

import torch  # noqa: E402

from entity_decode import entity_decode, score_candidate  # noqa: E402
from eval_entities import retention  # noqa: E402
from sample import load_model  # noqa: E402

ckpt_hits = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
assert ckpt_hits, "attach subhajitlucky/kalia-v020-ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = load_model(ckpt_hits[0], device)
enc = tiktoken.get_encoding("gpt2")

stories = json.loads(open("eval/ttt_probe_stories.json").read())
PROMPT_TOKENS = 768
GEN_TOKENS = 256
N = 8

rows = []
for i, story in enumerate(stories):
    ids = enc.encode_ordinary(story["prompt"])[:PROMPT_TOKENS]
    prompt_text = enc.decode(ids)
    greedy = entity_decode(
        model, enc, prompt_text, n=1, max_new_tokens=GEN_TOKENS,
        temperature=1e-6, top_k=None, seed=7 + i, device=device,
    )
    sampled = entity_decode(
        model, enc, prompt_text, n=N, max_new_tokens=GEN_TOKENS,
        temperature=0.8, top_k=200, seed=7 + i, device=device,
    )
    logprobs = [row["logprob"] for row in sampled.scores]
    lp_index = max(range(N), key=lambda j: logprobs[j])
    rows.append(
        {
            "id": story["id"],
            "greedy": score_candidate(prompt_text, greedy.texts[0], greedy.scores[0]["logprob"]),
            "logprob_pick": score_candidate(
                prompt_text, sampled.texts[lp_index], logprobs[lp_index]
            ),
            "entity_pick": score_candidate(
                prompt_text, sampled.texts[sampled.chosen], sampled.scores[sampled.chosen]["logprob"]
            ),
            "reference": score_candidate(prompt_text, story["reference_continuation"]),
            "entity_pick_index": sampled.chosen,
            "logprob_pick_index": lp_index,
            "same_pick": int(sampled.chosen == lp_index),
        }
    )
    print(f"{story['id']}: greedy {rows[-1]['greedy']['retention']:.2f} | "
          f"logprob {rows[-1]['logprob_pick']['retention']:.2f} | "
          f"entity {rows[-1]['entity_pick']['retention']:.2f} | "
          f"reference {rows[-1]['reference']['retention']:.2f}")

selectors = ("greedy", "logprob_pick", "entity_pick", "reference")
aggregate = {
    name: {
        "retention": sum(r[name]["retention"] for r in rows) / len(rows),
        "mean_span": sum(r[name]["mean_span"] for r in rows) / len(rows),
    }
    for name in selectors
}
aggregate["same_pick_count"] = sum(r["same_pick"] for r in rows)

print("\n== aggregate (20 stories, prompt 768, gen 256) ==")
for name in selectors:
    print(f"  {name:>12s}: retention {aggregate[name]['retention']:.3f} | "
          f"span {aggregate[name]['mean_span']:.3f}")
print(f"  entity pick == logprob pick: {aggregate['same_pick_count']}/{len(rows)}")

report = {
    "prompt_tokens": PROMPT_TOKENS,
    "gen_tokens": GEN_TOKENS,
    "n": N,
    "stories": rows,
    "aggregate": aggregate,
}
with open("/kaggle/working/entity_decode.json", "w") as fh:
    json.dump(report, fh, indent=2)
print("\nwrote /kaggle/working/entity_decode.json")
