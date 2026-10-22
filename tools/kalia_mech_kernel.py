"""Kaggle GPU probe: X32 — the entity-selection circuit.

Registered as ``docs/preregistrations/2026-10-22-X32-entity-selection-circuit.md``.
The entity holdout showed the model's candidate pool contains a better
continuation than greedy (oracle 0.0618 vs 0.0170 on the unseen half), while
every cheap selector fails. This probe asks whether a specific attention head
carries the wrong-name signal, and whether ablating it improves the continuation.

Design: 20 frozen stories, 768 prompt tokens, 256 generated, 8 samples, one
greedy draw. Generate the pools with the unmodified model; then run a manual
attention path on prompt + every candidate, recording per-head attention to the
prompt's entity tokens. Ablate the top three heads and the random-three control,
and score second-half entity retention. A reference-text NLL guard catches a
model that has been damaged into echoing names.

Outputs: ``x32_summary.json``, ``x32_heads.json``, ``run_manifest.json``.
"""

import glob
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ARM = "entity-circuit"  # staging pins this
KERNEL_ID = "unknown"  # staging pins this
KERNEL_REVISION = "unknown"  # staging pins this

PROMPT_TOKENS = 768
GEN_TOKENS = 256
N_SAMPLES = 8
TEMPERATURE = 0.8
TOP_K = 200
TOP_HEADS = 3
H2_BAR = 0.010
H1_ENRICH = 3.0
H1_TOP3_ENRICH = 2.0
H4_NLL_RATIO = 0.05


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


def find_subsequence(ids: list[int], sub: list[int]) -> list[int]:
    if not sub:
        return []
    return [i for i in range(len(ids) - len(sub) + 1)
            if ids[i:i + len(sub)] == sub]


def entity_positions(prompt_ids: list[int], prompt_text: str, enc) -> list[int]:
    from eval_entities import extract_entities

    positions: set[int] = set()
    for entity in extract_entities(prompt_text):
        for form in (entity, " " + entity):
            sub = enc.encode_ordinary(form)
            positions.update(find_subsequence(prompt_ids, sub))
    return sorted(positions)


def paired_bootstrap(a: list[float], b: list[float], iters: int = 5000) -> dict:
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(1234)
    means = []
    for _ in range(iters):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "mean": sum(diffs) / n,
        "lo": means[int(0.025 * iters)],
        "hi": means[int(0.975 * iters)],
        "excludes_zero": means[int(0.025 * iters)] > 0 or means[int(0.975 * iters)] < 0,
    }


# --- code and checkpoint ----------------------------------------------------
hits = sorted(glob.glob("/kaggle/input/**/model.py", recursive=True))
preferred = [p for p in hits if "/kalia-code-dev/" in p]
assert preferred, "attach kalia-code-dev; other model.py matches are stale: " + str(hits)
code_root = os.path.dirname(preferred[0])
work = "/kaggle/working/kalia"
if not os.path.exists(work):
    shutil.copytree(code_root, work)
os.chdir(work)
sys.path.insert(0, work)
for req in ("model.py", "eval_entities.py", "eval_reversibility.py",
            "eval/ttt_probe_stories.json"):
    assert os.path.exists(req), f"{req} missing from the code snapshot"
print("code:", code_root)

ckpts = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
assert ckpts, "attach subhajitlucky/kalia-v020-ckpt"
preferred_ckpt = [p for p in ckpts if "/kalia-v020-ckpt/" in p]
ckpt = (preferred_ckpt or ckpts)[0]
print("checkpoint:", ckpt)

import tiktoken  # noqa: E402
from eval_entities import retention  # noqa: E402
from eval_reversibility import load_model  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = load_model(ckpt, device)
torch.set_grad_enabled(False)
assert model.cfg.n_loops == 1, "this probe assumes one pass through the blocks"
enc = tiktoken.get_encoding("gpt2")
stories = json.loads(Path("eval/ttt_probe_stories.json").read_text())
print("stories:", len(stories))

# --- generate the candidate pools with the unmodified (SDPA) model ---------
def sample_candidates(prompt_ids, n, seed):
    gen = torch.Generator(device=prompt_ids.device).manual_seed(seed)
    out = []
    for _ in range(n):
        seq = model.generate(
            prompt_ids.unsqueeze(0), max_new_tokens=GEN_TOKENS,
            temperature=TEMPERATURE, top_k=TOP_K, generator=gen,
        )
        out.append(seq[0, prompt_ids.numel():].cpu())
    return out


@torch.no_grad()
def greedy(prompt_ids):
    seq = model.generate(
        prompt_ids.unsqueeze(0), max_new_tokens=GEN_TOKENS,
        temperature=1e-6, top_k=None,
    )
    return seq[0, prompt_ids.numel():].cpu()


@torch.no_grad()
def second_half_retention(prompt_text: str, seq: torch.Tensor) -> float:
    return retention(prompt_text, enc.decode(seq[GEN_TOKENS // 2:].tolist()))


candidates = {}
for i, story in enumerate(stories):
    ids = enc.encode_ordinary(story["prompt"])[:PROMPT_TOKENS]
    prompt_text = enc.decode(ids)
    prompt_ids = torch.tensor(ids, dtype=torch.long, device=device)
    candidates[story["id"]] = {
        "prompt_ids": ids,
        "prompt_text": prompt_text,
        "greedy": greedy(prompt_ids).tolist(),
        "samples": [s.tolist() for s in sample_candidates(prompt_ids, N_SAMPLES, 7 + i)],
    }
    print(f"generated {story['id']} ({len(ids)} prompt tokens)")

# --- manual attention path with capture and head ablation ------------------
CAPTURE = {"on": False, "qstart": 0, "entities": [], "stats": None}
ABLATE: set[tuple[int, int]] = set()


def init_stats():
    return {
        "entity": torch.zeros(model.cfg.n_layer, model.cfg.n_head),
        "recent": torch.zeros(model.cfg.n_layer, model.cfg.n_head),
        "entropy": torch.zeros(model.cfg.n_layer, model.cfg.n_head),
        "uniform": torch.zeros(model.cfg.n_layer, model.cfg.n_head),
        "count": 0,
    }


def summarize(layer: int, probs: torch.Tensor, qstart: int, ent_pos: list[int]) -> None:
    stats = CAPTURE["stats"]
    t = probs.size(-1)
    q = probs[0, :, qstart:, :]  # (H, Q, T)
    if q.numel() == 0 or not ent_pos:
        return
    ent = q[:, :, ent_pos].sum(-1).mean(-1)  # (H,)
    recent_start = max(0, qstart - 32)
    recent = q[:, :, recent_start:qstart].sum(-1).mean(-1)
    entropy = -(q.clamp_min(1e-9).log() * q).sum(-1).mean(-1)
    stats["entity"][layer] += ent.cpu()
    stats["recent"][layer] += recent.cpu()
    stats["entropy"][layer] += entropy.cpu()
    stats["uniform"][layer] += len(ent_pos) / t
    stats["count"] += 1


def patched_forward(self, x, attn_mask=None):
    b, t, c = x.shape
    q, k, v = self.qkv(x).split(
        [self.n_head * self.head_dim, self.n_kv_head * self.head_dim,
         self.n_kv_head * self.head_dim], dim=2,
    )
    q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
    k = k.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
    v = v.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
    if self.q_norm is not None:
        q = self.q_norm(q)
        k = self.k_norm(k)
    if self.use_rope:
        from model import apply_rope
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
    if self.n_kv_head != self.n_head:
        repeat = self.n_head // self.n_kv_head
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    probs = F.softmax(scores, dim=-1)
    if CAPTURE["on"]:
        summarize(self._x32_layer, probs, CAPTURE["qstart"], CAPTURE["entities"])
    for layer, head in ABLATE:
        if layer == self._x32_layer:
            probs[:, head, :, :] = 0.0
    y = probs @ v
    y = y.transpose(1, 2).contiguous().view(b, t, c)
    return self.proj(y)


for layer, block in enumerate(model.blocks):
    block.attn._x32_layer = layer
    block.attn.forward = patched_forward.__get__(block.attn)

# --- attention map on every prompt + candidate ----------------------------
stats = init_stats()
CAPTURE.update({"on": True, "stats": stats})
analyzed = 0
for story in stories:
    row = candidates[story["id"]]
    prompt_ids = row["prompt_ids"]
    ent_pos = entity_positions(prompt_ids, row["prompt_text"], enc)
    if not ent_pos:
        print(f"skip {story['id']}: no entity token in prompt")
        continue
    for seq in [row["greedy"]] + row["samples"]:
        full = prompt_ids + seq
        full = full[:model.cfg.context_len]
        if len(full) <= len(prompt_ids) + 1:
            continue
        CAPTURE.update({"qstart": len(prompt_ids), "entities": ent_pos})
        x = torch.tensor(full, dtype=torch.long, device=device).unsqueeze(0)
        model(x)
        analyzed += 1
CAPTURE["on"] = False
assert analyzed >= 100, f"only {analyzed} prompt+candidate passes analyzed"

count = max(1, stats["count"])
entity_mean = stats["entity"] / count
recent_mean = stats["recent"] / count
entropy_mean = stats["entropy"] / count
uniform_mean = (stats["uniform"] / count).clamp_min(1e-6)
enrichment = entity_mean / uniform_mean
heads = []
for layer in range(model.cfg.n_layer):
    for head in range(model.cfg.n_head):
        heads.append({
            "layer": layer, "head": head,
            "entity_mass": float(entity_mean[layer, head]),
            "uniform_mass": float(uniform_mean[layer, head]),
            "enrichment": float(enrichment[layer, head]),
            "recent_mass": float(recent_mean[layer, head]),
            "entropy": float(entropy_mean[layer, head]),
        })
heads.sort(key=lambda h: h["enrichment"], reverse=True)
top_heads = [(h["layer"], h["head"]) for h in heads[:TOP_HEADS]]
pool = [h for h in heads if (h["layer"], h["head"]) not in top_heads]
random_heads = [(h["layer"], h["head"]) for h in
                random.Random(1234).sample(pool, TOP_HEADS)]
print("top heads:", heads[:TOP_HEADS])
print("random control:", random_heads)

# --- ablation: greedy generation and reference NLL -----------------------
def generate_with_heads(prompt_ids, heads):
    ABLATE.clear()
    ABLATE.update(heads)
    out = greedy(prompt_ids)
    ABLATE.clear()
    return out


@torch.no_grad()
def reference_nll(prompt_ids, ref_ids, heads):
    ABLATE.clear()
    ABLATE.update(heads)
    ids = (prompt_ids + ref_ids)[:model.cfg.context_len]
    if len(ids) < 2:
        ABLATE.clear()
        return None
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    _, loss = model(x, y)
    ABLATE.clear()
    return float(loss.item())


rows = []
for story in stories:
    row = candidates[story["id"]]
    prompt_ids = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
    ref_ids = enc.encode_ordinary(story["reference_continuation"])
    greedy_ret = second_half_retention(row["prompt_text"],
                                       torch.tensor(row["greedy"]))
    top_ret = second_half_retention(
        row["prompt_text"], generate_with_heads(prompt_ids, top_heads))
    rand_ret = second_half_retention(
        row["prompt_text"], generate_with_heads(prompt_ids, random_heads))
    base_nll = reference_nll(row["prompt_ids"], ref_ids, [])
    top_nll = reference_nll(row["prompt_ids"], ref_ids, top_heads)
    rows.append({
        "id": story["id"],
        "greedy": greedy_ret,
        "top": top_ret,
        "random": rand_ret,
        "base_nll": base_nll,
        "top_nll": top_nll,
    })
    print(f"{story['id']}: greedy {greedy_ret:.3f} top {top_ret:.3f} "
          f"random {rand_ret:.3f}")

top_vs_greedy = paired_bootstrap([r["top"] for r in rows],
                                 [r["greedy"] for r in rows])
rand_vs_greedy = paired_bootstrap([r["random"] for r in rows],
                                  [r["greedy"] for r in rows])
top_vs_random = paired_bootstrap([r["top"] for r in rows],
                                 [r["random"] for r in rows])
base_nll = sum(r["base_nll"] for r in rows if r["base_nll"] is not None)
top_nll = sum(r["top_nll"] for r in rows if r["top_nll"] is not None)
n_nll = sum(1 for r in rows if r["base_nll"] is not None)
nll_delta = (top_nll - base_nll) / max(1, n_nll)
nll_ratio = (top_nll / base_nll - 1.0) if base_nll else 0.0

h1 = heads[0]["enrichment"] >= H1_ENRICH and all(
    h["enrichment"] >= H1_TOP3_ENRICH for h in heads[:TOP_HEADS])
h2 = top_vs_greedy["mean"] >= H2_BAR and top_vs_greedy["excludes_zero"]
h3 = rand_vs_greedy["mean"] < H2_BAR
h4 = nll_ratio <= H4_NLL_RATIO
verdict = {
    "H1": h1, "H2": h2, "H3": h3, "H4": h4,
    "H1_top_enrichment": heads[0]["enrichment"],
    "H1_top3_enrichment": [h["enrichment"] for h in heads[:TOP_HEADS]],
    "H2_top_minus_greedy": top_vs_greedy,
    "H3_random_minus_greedy": rand_vs_greedy,
    "H2_top_minus_random": top_vs_random,
    "H4_nll_base": base_nll / max(1, n_nll),
    "H4_nll_top": top_nll / max(1, n_nll),
    "H4_nll_delta": nll_delta,
    "H4_nll_ratio": nll_ratio,
    "decision": (
        "causal head evidence; replicate before a capability claim"
        if (h1 and h2 and h3 and h4) else
        "no single-head intervention; report the null"
    ),
}
print("\nX32 VERDICT")
for key in ("H1", "H2", "H3", "H4"):
    print(f"  {key}: {'PASS' if verdict[key] else 'FAIL'}")
print(f"  top enrichment: {heads[0]['enrichment']:.2f}x  "
      f"top3: {[round(x, 2) for x in verdict['H1_top3_enrichment']]}")
print(f"  top - greedy: {top_vs_greedy['mean']:+.4f} "
      f"[{top_vs_greedy['lo']:+.4f}, {top_vs_greedy['hi']:+.4f}]")
print(f"  random - greedy: {rand_vs_greedy['mean']:+.4f} "
      f"[{rand_vs_greedy['lo']:+.4f}, {rand_vs_greedy['hi']:+.4f}]")
print(f"  NLL ratio: {nll_ratio:+.3%}")
print("  decision:", verdict["decision"])

summary = {
    "experiment": ARM,
    "stories": len(rows),
    "candidates_per_story": N_SAMPLES,
    "attention_passes": analyzed,
    "top_heads": heads[:10],
    "top_three": top_heads,
    "random_three": random_heads,
    "rows": rows,
    "verdict": verdict,
}
write_json("/kaggle/working/x32_summary.json", summary)
write_json("/kaggle/working/x32_heads.json", heads)
write_json("/kaggle/working/run_manifest.json", {
    "experiment": ARM,
    "kernel_id": KERNEL_ID,
    "kernel_revision": KERNEL_REVISION,
    "code_snapshot": code_root,
    "code_fingerprint": code_fingerprint(code_root),
    "checkpoint": {"path": ckpt, "sha256": sha256_file(ckpt)},
    "stories_sha256": sha256_file("eval/ttt_probe_stories.json"),
    "prompt_tokens": PROMPT_TOKENS,
    "gen_tokens": GEN_TOKENS,
    "samples": N_SAMPLES,
    "top_heads": TOP_HEADS,
    "bars": {"H1": H1_ENRICH, "H1_top3": H1_TOP3_ENRICH,
             "H2": H2_BAR, "H4_nll_ratio": H4_NLL_RATIO},
})
print("wrote x32_summary.json, x32_heads.json, run_manifest.json")
