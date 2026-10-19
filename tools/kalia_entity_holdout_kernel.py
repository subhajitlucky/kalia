"""Kaggle kernel: does entity-verified decoding help, or does it grade its own homework?

The recorded result is that entity-pick triples greedy retention (0.0907 against
0.0296, +0.061) and lands near the human reference (0.0978). It is the project's
one inference-time win and the basis for the memory-only rung.

It is also close to circular. `entity_decode.select_best_index` picks the
candidate maximising `(retention, mean_span, logprob)`, where `retention` comes
from `eval_entities` — and `eval_entities.retention` is the metric the kernel then
reports. The measurement is "a selector that maximises X achieves a high X".

That is not worthless -- it says the good continuation is *in the sample set* and
the model's own probabilities cannot find it (logprob-pick scores 0.0353). But it
is a statement about **coverage**, not about the rule, and it is the same
self-reference the project bans elsewhere.

**The test that breaks it.** Split every generated continuation at its midpoint.
The selector sees only the **first half**; the metric is retention measured on
the **second half**, which the selector never saw.

    select on   retention(prompt, first_half)
    report      retention(prompt, second_half)

If selecting on the first half raises retention in the second half above greedy,
entities genuinely persist further than where the rule looked. If it does not,
the recorded 3x was the selector grading its own homework.

Reported for every story, so the comparison is paired:

    greedy        one deterministic draw
    logprob_pick  the model's own ranking (the recorded negative control)
    half_pick     first-half rule, scored on the unseen second half  <- the test
    oracle        best second-half retention among the 8             <- coverage
    reference     the human continuation                             <- ceiling
    full_pick     the recorded selector, scored on the full text     <- the 3x

CPU on purpose. It is a measurement, not a training run: it costs no GPU quota,
and it does not compete for the two batch GPU sessions -- which were both held by
stale scheduler entries when this was first pushed, and which queued an earlier
X17 attempt for 25 minutes for the same reason. 20 stories x 8 samples x 256
tokens is a few tens of minutes of CPU.
"""

import glob
import hashlib
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ARM = "entity-holdout"  # staging pins this; it names the experiment and its outputs
KERNEL_ID = "unknown"  # staging pins, from the committed metadata
KERNEL_REVISION = "unknown"  # staging pins, from git HEAD

PROMPT_TOKENS = 768
GEN_TOKENS = 256
N = 8
BOOTSTRAP = 2000
TEMPERATURE = 0.8
TOP_K = 200


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_run_manifest(out_path, **fields) -> dict:
    fields.update(kernel_id=KERNEL_ID, kernel_revision=KERNEL_REVISION,
                  recorded=datetime.now(timezone.utc).isoformat())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(fields, fh, indent=2, sort_keys=True)
    print("run manifest: " + json.dumps(fields, sort_keys=True))
    return fields


# --- the code, resolved the same way every kernel resolves it ----------------
hits = sorted(glob.glob("/kaggle/input/**/entity_decode.py", recursive=True))
preferred = [p for p in hits if "/kalia-code-dev/" in p]
assert preferred, (
    "kalia-code-dev not attached or lacks entity_decode.py; other matches "
    "(do NOT trust them, they are stale bundles): " + str(hits)
)
code_root = os.path.dirname(preferred[0])
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("entity_decode.py", "eval_entities.py", "best_of_n.py", "model.py",
            "sample.py", "eval/ttt_probe_stories.json"):
    assert os.path.exists(req), f"{req} missing from the copied code snapshot"
print("code:", code_root)

import torch  # noqa: E402

from best_of_n import continuation_logprob  # noqa: E402
from eval_entities import retention  # noqa: E402
from sample import load_model  # noqa: E402

ckpt_hits = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
assert ckpt_hits, "attach subhajitlucky/kalia-v020-ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = load_model(ckpt_hits[0], device)

import tiktoken  # noqa: E402

enc = tiktoken.get_encoding("gpt2")
stories = json.loads(open("eval/ttt_probe_stories.json").read())

# Fail fast on tokenizer API misuse, before an hour of measurement. The first
# version of this kernel called .tolist() on encode_ordinary's return value --
# it is a plain list, not a tensor -- and died after completing all of story 0's
# work. One assertion costs nothing and catches the whole class.
_probe = enc.encode_ordinary("Once upon a time there was a cat named Max.")
assert isinstance(_probe, list) and len(_probe) > 2, "encode_ordinary is not returning a list"
_half = len(_probe) // 2
assert enc.decode(_probe[_half:]), "decoding a list slice of ids failed"
assert enc.decode(torch.tensor(_probe)[_half:].tolist()), "decoding a tensor slice failed"
print("tokenizer probe ok:", repr(enc.decode(_probe[_half:])))


def sample_candidates(prompt_ids, n, seed, temperature, top_k):
    """n continuations, returned as token tensors so they can be split."""
    gen = torch.Generator(device=prompt_ids.device).manual_seed(int(seed))
    outs = []
    for _ in range(n):
        out = model.generate(
            prompt_ids.unsqueeze(0),
            max_new_tokens=GEN_TOKENS,
            temperature=temperature,
            top_k=top_k,
            generator=gen,
        )
        outs.append(out[0, prompt_ids.numel():])
    return outs


def paired_bootstrap(a: list[float], b: list[float], iters: int = BOOTSTRAP):
    """95% CI for mean(a - b) by resampling stories. Positive means a wins."""
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(1234)
    means = []
    n = len(diffs)
    for _ in range(iters):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "mean": sum(diffs) / n,
        "lo": means[int(0.025 * iters)],
        "hi": means[int(0.975 * iters)],
        "excludes_zero": means[int(0.025 * iters)] > 0 or means[int(0.975 * iters)] < 0,
    }


rows = []
for i, story in enumerate(stories):
    ids = enc.encode_ordinary(story["prompt"])[:PROMPT_TOKENS]
    prompt_text = enc.decode(ids)
    prompt_ids = torch.tensor(ids, dtype=torch.long, device=device)

    greedy_seq = sample_candidates(prompt_ids, 1, 7 + i, 1e-6, None)[0]
    cands = sample_candidates(prompt_ids, N, 7 + i, TEMPERATURE, TOP_K)

    half = GEN_TOKENS // 2
    cand = []
    for seq in cands:
        first = enc.decode(seq[:half].tolist())
        second = enc.decode(seq[half:].tolist())
        full = enc.decode(seq.tolist())
        cand.append({
            "first_retention": retention(prompt_text, first),
            "holdout_retention": retention(prompt_text, second),
            "full_retention": retention(prompt_text, full),
            "logprob": continuation_logprob(model, prompt_ids, seq)[1],
        })

    greedy_second = enc.decode(greedy_seq[half:].tolist())
    greedy_full = enc.decode(greedy_seq.tolist())
    # encode_ordinary returns a python list, not a tensor: the first version of
    # this line called .tolist() on it and died after doing all of story 0's work.
    ref_second = enc.decode(
        enc.encode_ordinary(story["reference_continuation"])[half:]
    )

    by_half = max(range(N), key=lambda j: cand[j]["first_retention"])
    by_logprob = max(range(N), key=lambda j: cand[j]["logprob"])
    by_full = max(range(N), key=lambda j: cand[j]["full_retention"])
    by_oracle = max(range(N), key=lambda j: cand[j]["holdout_retention"])

    rows.append({
        "id": story["id"],
        "greedy": {"holdout": retention(prompt_text, greedy_second),
                   "full": retention(prompt_text, greedy_full)},
        "logprob_pick": {"holdout": cand[by_logprob]["holdout_retention"],
                         "full": cand[by_logprob]["full_retention"]},
        "half_pick": {"holdout": cand[by_half]["holdout_retention"],
                      "full": cand[by_half]["full_retention"],
                      "index": by_half},
        "full_pick": {"holdout": cand[by_full]["holdout_retention"],
                      "full": cand[by_full]["full_retention"],
                      "index": by_full},
        "oracle": {"holdout": cand[by_oracle]["holdout_retention"]},
        "reference": {"holdout": retention(prompt_text, ref_second),
                      "full": retention(prompt_text, story["reference_continuation"])},
        "spread_of_holdout": max(c["holdout_retention"] for c in cand)
        - min(c["holdout_retention"] for c in cand),
    })
    print(f"{story['id']}: greedy {rows[-1]['greedy']['holdout']:.3f} | "
          f"half_pick {rows[-1]['half_pick']['holdout']:.3f} | "
          f"oracle {rows[-1]['oracle']['holdout']:.3f} | "
          f"reference {rows[-1]['reference']['holdout']:.3f}")
    # Written per story, not once at the end. The first run died on a one-line
    # bug in the *last* computation of story 0 and threw away everything it had
    # measured getting there -- the same shape as the 2x2 kernel, which failed
    # while printing its summary with all eight cells already computed. A long
    # measurement should not be able to lose itself to a typo in its own report.
    with open("/kaggle/working/entity_holdout.partial.json", "w") as fh:
        json.dump({"stories": rows}, fh, indent=2)

names = ("greedy", "logprob_pick", "half_pick", "full_pick", "oracle", "reference")
agg = {n: sum(r[n]["holdout"] for r in rows) / len(rows) for n in names}
agg_full = {n: sum(r[n]["full"] for r in rows) / len(rows)
            for n in ("greedy", "logprob_pick", "half_pick", "full_pick", "reference")}

print(f"\n== holdout retention (selector never saw this half), {len(rows)} stories ==")
for n in names:
    print(f"  {n:>13s}: {agg[n]:.4f}")
print("\n== full-continuation retention (what the recorded run reported) ==")
for n in agg_full:
    print(f"  {n:>13s}: {agg_full[n]:.4f}")

tests = {
    "half_pick_minus_greedy_holdout": paired_bootstrap(
        [r["half_pick"]["holdout"] for r in rows], [r["greedy"]["holdout"] for r in rows]),
    "half_pick_minus_logprob_holdout": paired_bootstrap(
        [r["half_pick"]["holdout"] for r in rows], [r["logprob_pick"]["holdout"] for r in rows]),
    "full_pick_minus_greedy_full": paired_bootstrap(
        [r["full_pick"]["full"] for r in rows], [r["greedy"]["full"] for r in rows]),
}
print("\n== paired bootstrap, 95% CI over stories ==")
for name, t in tests.items():
    verdict = "excludes zero" if t["excludes_zero"] else "INCLUDES ZERO"
    print(f"  {name}: {t['mean']:+.4f} [{t['lo']:+.4f}, {t['hi']:+.4f}]  {verdict}")

# Invariants that a silent logic error would break. The oracle picks the best
# held-out retention by construction, so anything scoring above it means the
# selection or the splitting is wrong -- a wrong number rather than a crash,
# which is the failure mode this project keeps finding.
for r in rows:
    assert r["oracle"]["holdout"] >= r["half_pick"]["holdout"] - 1e-12, r["id"]
    assert r["oracle"]["holdout"] >= r["full_pick"]["holdout"] - 1e-12, r["id"]
    assert r["oracle"]["holdout"] >= r["logprob_pick"]["holdout"] - 1e-12, r["id"]
    assert 0.0 <= r["greedy"]["holdout"] <= 1.0 and 0.0 <= r["half_pick"]["holdout"] <= 1.0
print(f"invariants ok on {len(rows)} stories (oracle bounds every selector)")

report = {
    "prompt_tokens": PROMPT_TOKENS, "gen_tokens": GEN_TOKENS, "n": N,
    "holdout_split": GEN_TOKENS // 2,
    "temperature": TEMPERATURE, "top_k": TOP_K, "bootstrap_iters": BOOTSTRAP,
    "holdout_aggregate": agg, "full_aggregate": agg_full,
    "tests": tests, "stories": rows,
}
with open("/kaggle/working/entity_holdout.json", "w") as fh:
    json.dump(report, fh, indent=2)

write_run_manifest(
    "/kaggle/working/run_manifest.json",
    experiment=ARM,
    stories=len(rows),
    samples_per_story=N,
    holdout_split=GEN_TOKENS // 2,
    code_snapshot=code_root,
    code_fingerprint=hashlib.sha256(
        b"".join(sorted(p.read_bytes() for p in Path(code_root).glob("*.py")))
    ).hexdigest(),
    checkpoint={"path": ckpt_hits[0], "sha256": sha256_file(ckpt_hits[0])},
    stories_sha256=sha256_file("eval/ttt_probe_stories.json"),
)
print("wrote /kaggle/working/entity_holdout.json")
