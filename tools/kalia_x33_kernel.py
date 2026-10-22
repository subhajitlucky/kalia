"""Kaggle GPU probe: X33 — does the midpoint state know whether the entity will survive?

Registered as ``docs/preregistrations/2026-10-22-X33-linear-readout.md``.
X32 found entity attention concentrated in layer-9 heads, but ablating them hurt
retention. X33 tests the readout directly: show the model the prompt plus only
the first 128 generated tokens, capture the layer-9 residual at the midpoint, and
ask whether a linear probe can predict the candidate's second-half entity
retention on a story it has never seen.

Outputs: ``x33_summary.json``, ``x33_features.npz``, ``run_manifest.json``.
"""

import glob
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ARM = "linear-readout"  # staging pins this
KERNEL_ID = "unknown"  # staging pins this
KERNEL_REVISION = "unknown"  # staging pins this

PROMPT_TOKENS = 768
GEN_TOKENS = 256
HALF = GEN_TOKENS // 2
N_SAMPLES = 8
TEMPERATURE = 0.8
TOP_K = 200
LAYER = 9
ALPHA = 10.0
P1_BAR = 0.010
P3_BAR = 0.010
SHUFFLES = 20


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


def ridge_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                  alpha: float = ALPHA) -> np.ndarray:
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Xc = (Xtr - mu) / sd
    Xt = (Xte - mu) / sd
    ym = float(ytr.mean())
    yc = ytr - ym
    d = Xc.shape[1]
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d, dtype=Xc.dtype), Xc.T @ yc)
    return Xt @ w + ym


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
            "best_of_n.py", "eval/ttt_probe_stories.json"):
    assert os.path.exists(req), f"{req} missing from the code snapshot"
print("code:", code_root)

ckpts = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
assert ckpts, "attach subhajitlucky/kalia-v020-ckpt"
preferred_ckpt = [p for p in ckpts if "/kalia-v020-ckpt/" in p]
ckpt = (preferred_ckpt or ckpts)[0]
print("checkpoint:", ckpt)

import tiktoken  # noqa: E402
from best_of_n import continuation_logprob  # noqa: E402
from eval_entities import retention  # noqa: E402
from eval_reversibility import load_model  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = load_model(ckpt, device)
torch.set_grad_enabled(False)
assert model.cfg.n_layer == 10, "X33 expects the 10-layer v0.2.0 model"
enc = tiktoken.get_encoding("gpt2")
stories = json.loads(Path("eval/ttt_probe_stories.json").read_text())
print("stories:", len(stories))

# --- generation -------------------------------------------------------------
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


# --- capture the layer-9 residual at the midpoint --------------------------
CAPTURE = {"on": False, "qstart": 0, "feat": None}


def capture_hook(module, inputs, output):
    if not CAPTURE["on"]:
        return
    x = output[0] if isinstance(output, tuple) else output
    cont = x[0, CAPTURE["qstart"]:, :]
    if cont.size(0) == 0:
        return
    feat = torch.cat([cont.mean(dim=0), cont[-1]], dim=0)
    CAPTURE["feat"] = feat.detach().cpu().numpy().astype(np.float32)


model.blocks[LAYER].register_forward_hook(capture_hook)


def midpoint_features(prompt_ids: list[int], cand: list[int]) -> np.ndarray:
    first = cand[:HALF]
    full = (prompt_ids + first)[:model.cfg.context_len]
    assert len(full) > len(prompt_ids) + 1, "candidate has no first-half tokens"
    CAPTURE.update({"on": True, "qstart": len(prompt_ids), "feat": None})
    x = torch.tensor(full, dtype=torch.long, device=device).unsqueeze(0)
    model(x)
    CAPTURE["on"] = False
    feat = CAPTURE["feat"]
    assert feat is not None and feat.shape == (2 * model.cfg.n_embd,), feat
    return feat


# --- build the candidate table ---------------------------------------------
features: list[np.ndarray] = []
second: list[list[float]] = []
first: list[list[float]] = []
logprobs: list[list[float]] = []
greedy_ret: list[float] = []
prompt_ids_list: list[list[int]] = []
prompt_texts: list[str] = []
ids_out: list[list[list[int]]] = []

for i, story in enumerate(stories):
    ids = enc.encode_ordinary(story["prompt"])[:PROMPT_TOKENS]
    prompt_text = enc.decode(ids)
    prompt_ids = torch.tensor(ids, dtype=torch.long, device=device)
    cands = [greedy(prompt_ids)] + sample_candidates(prompt_ids, N_SAMPLES, 7 + i)
    # The first draw is greedy; the eight samples are the candidate pool.
    pool = cands[1:]
    feats = np.stack([midpoint_features(ids, c.tolist()) for c in pool])
    second_row = [retention(prompt_text, enc.decode(c[HALF:].tolist())) for c in pool]
    first_row = [retention(prompt_text, enc.decode(c[:HALF].tolist())) for c in pool]
    logp_row = [continuation_logprob(model, prompt_ids, c.to(device))[1] for c in pool]
    features.append(feats)
    second.append(second_row)
    first.append(first_row)
    logprobs.append(logp_row)
    greedy_ret.append(retention(prompt_text, enc.decode(cands[0][HALF:].tolist())))
    prompt_ids_list.append(ids)
    prompt_texts.append(prompt_text)
    ids_out.append([c.tolist() for c in pool])
    print(f"{story['id']}: pool built ({feats.shape}); "
          f"greedy {greedy_ret[-1]:.3f}")

X = np.stack(features)          # (stories, candidates, 2 * d)
y = np.array(second)            # (stories, candidates)
half = np.array(first)
logs = np.array(logprobs)
greedy_arr = np.array(greedy_ret)
n_stories, n_cands = y.shape

# --- leave-one-story-out linear probe --------------------------------------
probe_ret: list[float] = []
probe_idx: list[int] = []
half_ret: list[float] = []
half_idx: list[int] = []
log_ret: list[float] = []
log_idx: list[int] = []
oracle_ret: list[float] = []

for i in range(n_stories):
    train = [j for j in range(n_stories) if j != i]
    Xtr = np.concatenate([X[j] for j in train], axis=0)
    ytr = np.concatenate([y[j] for j in train], axis=0)
    pred = ridge_predict(Xtr, ytr, X[i])
    pi = int(np.argmax(pred))
    hi = int(np.argmax(half[i]))
    li = int(np.argmax(logs[i]))
    oi = int(np.argmax(y[i]))
    probe_idx.append(pi)
    probe_ret.append(float(y[i, pi]))
    half_idx.append(hi)
    half_ret.append(float(y[i, hi]))
    log_idx.append(li)
    log_ret.append(float(y[i, li]))
    oracle_ret.append(float(y[i, oi]))
    print(f"{stories[i]['id']}: probe {y[i, pi]:.3f} greedy {greedy_arr[i]:.3f} "
          f"half {y[i, hi]:.3f} logp {y[i, li]:.3f} oracle {y[i, oi]:.3f}")

# --- shuffled-label control ------------------------------------------------
rng = np.random.default_rng(1234)
shuffled_means = []
for _ in range(SHUFFLES):
    vals = []
    for i in range(n_stories):
        train = [j for j in range(n_stories) if j != i]
        Xtr = np.concatenate([X[j] for j in train], axis=0)
        ytr = np.concatenate([y[j] for j in train], axis=0)
        ytr = rng.permutation(ytr)
        pred = ridge_predict(Xtr, ytr, X[i])
        vals.append(float(y[i, int(np.argmax(pred))]))
    shuffled_means.append(float(np.mean(vals)))
shuffled_mean = float(np.mean(shuffled_means))

# --- verdict ----------------------------------------------------------------
probe_minus_greedy = paired_bootstrap(probe_ret, greedy_arr.tolist())
probe_minus_half = paired_bootstrap(probe_ret, half_ret)
probe_minus_log = paired_bootstrap(probe_ret, log_ret)
shuffled_minus_greedy = shuffled_mean - float(greedy_arr.mean())
p1 = probe_minus_greedy["mean"] >= P1_BAR and probe_minus_greedy["excludes_zero"]
p2 = (probe_minus_half["mean"] >= 0) and (probe_minus_log["mean"] >= 0)
p3 = shuffled_minus_greedy < P3_BAR
verdict = {
    "P1": bool(p1), "P2": bool(p2), "P3": bool(p3),
    "probe_mean": float(np.mean(probe_ret)),
    "greedy_mean": float(greedy_arr.mean()),
    "half_mean": float(np.mean(half_ret)),
    "logprob_mean": float(np.mean(log_ret)),
    "oracle_mean": float(np.mean(oracle_ret)),
    "probe_minus_greedy": probe_minus_greedy,
    "probe_minus_half": probe_minus_half,
    "probe_minus_logprob": probe_minus_log,
    "shuffled_mean": shuffled_mean,
    "shuffled_minus_greedy": shuffled_minus_greedy,
    "decision": (
        "linear midpoint signal; replicate before a selector claim"
        if (p1 and p2 and p3) else
        "no linear midpoint signal; close the linear-selector track"
    ),
}
print("\nX33 VERDICT")
for key in ("P1", "P2", "P3"):
    print(f"  {key}: {'PASS' if verdict[key] else 'FAIL'}")
print(f"  probe {verdict['probe_mean']:.4f} | greedy {verdict['greedy_mean']:.4f} "
      f"| half {verdict['half_mean']:.4f} | logp {verdict['logprob_mean']:.4f} "
      f"| oracle {verdict['oracle_mean']:.4f}")
print(f"  probe - greedy: {probe_minus_greedy['mean']:+.4f} "
      f"[{probe_minus_greedy['lo']:+.4f}, {probe_minus_greedy['hi']:+.4f}]")
print(f"  probe - half: {probe_minus_half['mean']:+.4f}; "
      f"probe - logp: {probe_minus_log['mean']:+.4f}")
print(f"  shuffled - greedy: {shuffled_minus_greedy:+.4f}")
print("  decision:", verdict["decision"])

# --- archive ----------------------------------------------------------------
rows = []
for i, story in enumerate(stories):
    rows.append({
        "id": story["id"],
        "greedy": float(greedy_arr[i]),
        "probe": probe_ret[i],
        "probe_index": probe_idx[i],
        "half": half_ret[i],
        "half_index": half_idx[i],
        "logprob": log_ret[i],
        "logprob_index": log_idx[i],
        "oracle": oracle_ret[i],
    })
summary = {
    "experiment": ARM,
    "stories": n_stories,
    "candidates_per_story": n_cands,
    "layer": LAYER,
    "features": "mean(first-half) + last-token, 2*d",
    "alpha": ALPHA,
    "rows": rows,
    "verdict": verdict,
}
write_json("/kaggle/working/x33_summary.json", summary)
np.savez_compressed(
    "/kaggle/working/x33_features.npz",
    features=X, second=y, first=half, logprob=logs,
    greedy=greedy_arr, prompt_ids=np.array(prompt_ids_list, dtype=object),
    candidate_ids=np.array(ids_out, dtype=object),
)
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
    "half": HALF,
    "samples": N_SAMPLES,
    "layer": LAYER,
    "alpha": ALPHA,
    "shuffles": SHUFFLES,
    "bars": {"P1": P1_BAR, "P3": P3_BAR},
})
print("wrote x33_summary.json, x33_features.npz, run_manifest.json")
