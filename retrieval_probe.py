"""Does KALIA use what it reads? Retrieval utilization and distraction at 58M.

Companion to ``docs/research/2026-10-08-frontier-recheck.md``. arXiv 2603.11513
found that sub-7B models fail to use retrieved context even when the answer is
guaranteed to be in it (85-100% failure on questions they cannot answer alone),
and that any context destroys 42-100% of answers the model already knew. KALIA
is 58M - about a hundred times below the smallest model tested there.

This harness measures the same questions on a base (non-instruction-tuned)
language model:

1. **Fact utilization.** Synthetic facts built from invented family names, so
   the answers are unknown by construction. Conditions: none / BM25 / oracle.
   If the model can read, oracle accuracy beats the eight-place chance level by
   a wide margin. If it cannot, the retrieved passage is decoration.
2. **Distraction on natural text.** Held-out windows the model predicts well,
   scored bare / with relevant context / with irrelevant context. Bare to
   irrelevant is the distraction effect; bare to relevant is utilization where
   the context is real.
3. **Best-of-N against its upper bound.** On the oracle condition: greedy vs
   log-prob-selected vs majority vs coverage (the ceiling any selector could
   reach). The selector in ``best_of_n.py`` is measured, not assumed to help.

Deterministic: every random choice draws from a seeded ``random.Random``, every
sampler from a per-item ``torch.Generator``. The JSON output is the record; no
number from this file is typed into prose anywhere else.

Usage:
    python retrieval_probe.py --ckpt out/ckpt.pt --val-bin data/val.bin --out probe.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from best_of_n import continuation_logprob
from retrieval import BM25Index

NOUNS = ["amulet", "locket", "compass", "mirror", "key", "crown", "flask", "lantern"]
# Invented family names: the facts cannot be recalled from pretraining because
# these strings never appeared in it. That is what makes "none" the true
# unknown-question condition.
FAMILIES = ["Varley", "Osten", "Marro", "Quelch", "Drevan", "Ilmoor", "Sarn", "Veyra"]
PLACES = ["attic", "cellar", "garden", "harbor", "tower", "forest", "chapel", "island"]
DISTRACTORS_PER_POOL = 7


def single_token_words(encoder, words: list[str]) -> list[str]:
    """Keep only words the encoder turns into one token after a space."""
    kept = []
    for word in words:
        ids = encoder.encode_ordinary(" " + word)
        if len(ids) == 1:
            kept.append(word)
    return kept


def answer_id(encoder, place: str) -> int:
    return encoder.encode_ordinary(" " + place)[0]


def bootstrap_ci(values: list[float], seed: int = 1234, resamples: int = 2000) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean; deterministic under ``seed``.

    Single-pass numbers on 64 items need an error bar, especially for the
    paired distraction deltas. The items are fixed, so this is a bootstrap
    over the measured set, not over hypothetical reruns.
    """
    if not values:
        raise ValueError("no values to bootstrap")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def build_fact_items(encoder, n: int, seed: int) -> list[dict]:
    """Synthetic facts with single-token answers and a per-item distractor pool.

    Only the *answers* need to be single tokens (so rank is unambiguous). Nouns
    and family names may tokenize into several pieces and that is fine -- the
    first version filtered all three lists and the invented surnames (e.g.
    "Varley") left the family list empty, which failed only on the real encoder.
    """
    places = single_token_words(encoder, PLACES)
    families = [f for f in FAMILIES if f]
    nouns = [w for w in NOUNS if w]
    if not places:
        raise ValueError("encoder produced no usable single-token place words")
    rng = random.Random(seed)
    combos = [(noun, family, place) for noun in nouns for family in families for place in places]
    rng.shuffle(combos)
    chosen = combos[:n]
    sentences = [
        f"The {noun} of the {family} family is hidden in the {place}." for noun, family, place in chosen
    ]
    items: list[dict] = []
    for i, (noun, family, place) in enumerate(chosen):
        pool = [sentences[i]]
        others = [j for j in range(len(sentences)) if j != i]
        pool += [sentences[j] for j in rng.sample(others, min(DISTRACTORS_PER_POOL, len(others)))]
        order = list(range(len(pool)))
        rng.shuffle(order)
        pool = [pool[k] for k in order]
        true_index = order.index(0)
        items.append(
            {
                "noun": noun,
                "family": family,
                "place": place,
                "sentence": sentences[i],
                "prompt": f"The {noun} of the {family} family is hidden in the",
                "answer_id": answer_id(encoder, place),
                "pool": pool,
                "true_index": true_index,
            }
        )
    return items


def build_distraction_items(val_ids: np.ndarray, n: int, seed: int,
                            context_len: int = 64, prompt_len: int = 64,
                            target_len: int = 16) -> list[dict]:
    """Windows of real held-out text plus a mismatched context for each."""
    span = context_len + prompt_len + target_len
    if len(val_ids) < span * (n + 2):
        raise ValueError(f"val shard too short ({len(val_ids)}) for {n} windows of {span}")
    rng = random.Random(seed)
    starts = rng.sample(range(0, len(val_ids) - span), n)
    windows = []
    for start in starts:
        ids = val_ids[start : start + span].astype(np.int64)
        windows.append(
            {
                "context_ids": ids[:context_len].tolist(),
                "prompt_ids": ids[context_len : context_len + prompt_len].tolist(),
                "target_ids": ids[context_len + prompt_len :].tolist(),
            }
        )
    for i, window in enumerate(windows):
        window["irrelevant_context_ids"] = windows[(i + n // 2) % n]["context_ids"]
    return windows


@torch.no_grad()
def score_next_token(model, ids: list[int], answer: int) -> tuple[int, float, bool]:
    """Rank, log-probability and top-1 flag for one next-token answer."""
    device = next(model.parameters()).device
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x)
    last = logits[0, -1]
    logprob = float(F.log_softmax(last, dim=-1)[answer])
    rank = int((last > last[answer]).sum().item()) + 1
    return rank, logprob, rank == 1


@torch.no_grad()
def mean_target_nll(model, context_ids: list[int], prompt_ids: list[int],
                    target_ids: list[int]) -> float:
    """Teacher-forced mean NLL of the target tokens after context + prompt."""
    device = next(model.parameters()).device
    seq = context_ids + prompt_ids + target_ids
    x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x)
    start = len(context_ids) + len(prompt_ids) - 1
    logp = F.log_softmax(logits[0, start : start + len(target_ids)], dim=-1)
    targets = torch.tensor(target_ids, dtype=torch.long, device=device)
    return float(-logp.gather(1, targets.unsqueeze(1)).squeeze(1).mean())


def run_fact_conditions(model, encoder, items: list[dict], top_k: int = 3,
                        seed: int = 1234) -> dict:
    """The three retrieval conditions over the synthetic fact set."""
    out: dict = {"n": len(items), "places": len(PLACES)}
    for condition in ("none", "bm25", "oracle"):
        ranks, logprobs, flags = [], [], []
        bm25_hits, bm25_rr = 0, 0.0
        for item in items:
            context = ""
            if condition == "bm25":
                index = BM25Index(item["pool"])
                hits = index.retrieve(item["prompt"], top_k)
                retrieved = [item["pool"][i] for i, _ in hits]
                context = "\n".join(retrieved)
                positions = [pos for pos, (i, _) in enumerate(hits) if i == item["true_index"]]
                if positions:
                    bm25_hits += 1
                    bm25_rr += 1.0 / (positions[0] + 1)
            elif condition == "oracle":
                context = item["sentence"]
            text = f"{context}\n\n{item['prompt']}" if context else item["prompt"]
            rank, logprob, is_top1 = score_next_token(
                model, encoder.encode_ordinary(text), item["answer_id"]
            )
            ranks.append(rank)
            logprobs.append(logprob)
            flags.append(float(is_top1))
        lo, hi = bootstrap_ci(flags, seed)
        out[condition] = {
            "top1": sum(flags) / len(items),
            "top1_ci95": [lo, hi],
            "mrr": sum(1.0 / r for r in ranks) / len(items),
            "mean_logprob": sum(logprobs) / len(items),
            "per_item_top1": flags,
            "per_item_rank": ranks,
        }
        if condition == "bm25":
            out["bm25_hit_rate"] = bm25_hits / len(items)
            out["bm25_true_rank_mrr"] = bm25_rr / len(items)
    return out


def run_distraction_conditions(model, windows: list[dict], seed: int = 1234) -> dict:
    """Bare vs relevant vs irrelevant context on natural held-out text."""
    out: dict = {"n": len(windows)}
    per_condition: dict[str, list[float]] = {}
    for condition in ("bare", "relevant", "irrelevant"):
        nlls = []
        for window in windows:
            if condition == "bare":
                context: list[int] = []
            elif condition == "relevant":
                context = window["context_ids"]
            else:
                context = window["irrelevant_context_ids"]
            nlls.append(
                mean_target_nll(model, context, window["prompt_ids"], window["target_ids"])
            )
        per_condition[condition] = nlls
        lo, hi = bootstrap_ci(nlls, seed)
        out[condition] = {
            "mean_nll": sum(nlls) / len(nlls),
            "mean_nll_ci95": [lo, hi],
            "per_item_nll": nlls,
        }
    for name in ("relevant", "irrelevant"):
        deltas = [a - b for a, b in zip(per_condition[name], per_condition["bare"])]
        lo, hi = bootstrap_ci(deltas, seed)
        out[f"{name}_minus_bare"] = {"mean": sum(deltas) / len(deltas), "ci95": [lo, hi]}
    return out


@torch.no_grad()
def run_best_of_n(model, encoder, items: list[dict], samples: int, seed: int,
                  max_new_tokens: int = 4, temperature: float = 0.8,
                  top_k: int = 200) -> dict:
    """Greedy vs k-window selection vs majority vs coverage on the oracle condition.

    The k-sweep exists because the first measurement scored the whole
    continuation with mean log-probability and selected *worse* than greedy
    (0.547 vs 0.828): a fluent wrong continuation out-scores a correct first
    token that continues awkwardly. Scoring only the decision position (k=1) is
    the repair candidate, measured here against the same greedy baseline and the
    coverage ceiling.
    """
    device = next(model.parameters()).device
    ks = sorted({1, 2, max_new_tokens})
    flags: dict[str, list[float]] = {"greedy": [], "majority": [], "coverage": []}
    for k in ks:
        flags[f"sel_k{k}"] = []
    for i, item in enumerate(items):
        text = f"{item['sentence']}\n\n{item['prompt']}"
        prompt = torch.tensor(encoder.encode_ordinary(text), dtype=torch.long, device=device)
        answer = item["answer_id"]

        logits, _ = model(prompt.unsqueeze(0))
        flags["greedy"].append(float(int(logits[0, -1].argmax()) == answer))

        generator = torch.Generator(device=device).manual_seed(seed + i)
        continuations = []
        for _ in range(samples):
            out = model.generate(
                prompt.unsqueeze(0),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            continuations.append(out[0, prompt.numel() :])
        first_tokens = [int(cont[0]) for cont in continuations]
        correct_flags = [t == answer for t in first_tokens]
        flags["coverage"].append(float(any(correct_flags)))
        for k in ks:
            scores = [
                continuation_logprob(model, prompt, cont, max_tokens=k)[1]
                for cont in continuations
            ]
            flags[f"sel_k{k}"].append(
                float(correct_flags[max(range(samples), key=lambda j: scores[j])])
            )
        counts: dict[int, int] = {}
        for token in first_tokens:
            counts[token] = counts.get(token, 0) + 1
        majority = max(counts, key=lambda t: (counts[t], -first_tokens.index(t)))
        flags["majority"].append(float(majority == answer))
    n = len(items)
    out: dict = {"n": n, "samples": samples, "score_token_windows": ks}
    for name, values in flags.items():
        lo, hi = bootstrap_ci(values, seed)
        out[f"{name}_top1"] = sum(values) / n
        out[f"{name}_top1_ci95"] = [lo, hi]
        out[f"per_item_{name}"] = values
    out["coverage"] = out["coverage_top1"]
    out["selected_top1"] = out[f"sel_k{max_new_tokens}_top1"]
    out["selected_top1_ci95"] = out[f"sel_k{max_new_tokens}_top1_ci95"]
    for k in ks:
        deltas = [a - b for a, b in zip(flags[f"sel_k{k}"], flags["greedy"])]
        lo, hi = bootstrap_ci(deltas, seed)
        label = "selected" if k == max_new_tokens else f"sel_k{k}"
        out[f"{label}_minus_greedy"] = {"mean": sum(deltas) / n, "ci95": [lo, hi]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--val-bin", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--facts", type=int, default=64)
    parser.add_argument("--distract", type=int, default=48)
    parser.add_argument("--n", type=int, default=8, help="best-of-N samples per item")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    import tiktoken

    from sample import load_model

    device = torch.device(args.device)
    model = load_model(args.ckpt, device)
    encoder = tiktoken.get_encoding("gpt2")

    report: dict = {"seed": args.seed, "checkpoint": str(args.ckpt)}

    items = build_fact_items(encoder, args.facts, args.seed)
    fact = run_fact_conditions(model, encoder, items)
    report["fact_utilization"] = fact
    print("== fact utilization (unknown by construction) ==")
    print(f"  n={fact['n']}  chance top-1 over {fact['places']} places ~ {1 / fact['places']:.3f}")
    for condition in ("none", "bm25", "oracle"):
        c = fact[condition]
        lo, hi = c["top1_ci95"]
        print(f"  {condition:>7s}: top1 {c['top1']:.3f} [{lo:.3f}, {hi:.3f}] | "
              f"mrr {c['mrr']:.3f} | mean logprob {c['mean_logprob']:+.3f}")
    print(f"  bm25 retrieved the true passage in top-3: {fact['bm25_hit_rate']:.3f}")

    if args.val_bin is not None:
        val_ids = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
        windows = build_distraction_items(val_ids, args.distract, args.seed)
        distraction = run_distraction_conditions(model, windows, args.seed)
        report["distraction"] = distraction
        print("\n== distraction (natural held-out text, mean NLL of 16 tokens) ==")
        for condition in ("bare", "relevant", "irrelevant"):
            print(f"  {condition:>10s}: {distraction[condition]['mean_nll']:.4f}")
        for name in ("relevant_minus_bare", "irrelevant_minus_bare"):
            row = distraction[name]
            lo, hi = row["ci95"]
            print(f"  {name:>19s}: {row['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    bestn = run_best_of_n(model, encoder, items, args.n, args.seed, top_k=args.top_k)
    report["best_of_n"] = bestn
    print("\n== best-of-N on the oracle condition ==")
    print(f"  n={bestn['n']} samples={bestn['samples']}")
    for name in ("greedy", "majority"):
        lo, hi = bestn[f"{name}_top1_ci95"]
        print(f"  {name:>10s}  top1 {bestn[f'{name}_top1']:.3f} [{lo:.3f}, {hi:.3f}]")
    for k in bestn["score_token_windows"]:
        key = f"sel_k{k}_top1"
        lo, hi = bestn[f"{key}_ci95"]
        print(f"  sel(k={k})   top1 {bestn[key]:.3f} [{lo:.3f}, {hi:.3f}]")
    lo, hi = bestn["coverage_top1_ci95"]
    print(f"  coverage    top1 {bestn['coverage_top1']:.3f} [{lo:.3f}, {hi:.3f}]  (ceiling)")
    full = bestn["score_token_windows"][-1]
    for k in bestn["score_token_windows"]:
        label = "selected" if k == full else f"sel_k{k}"
        delta = bestn[f"{label}_minus_greedy"]
        print(f"  {label} - greedy: {delta['mean']:+.3f} "
              f"[{delta['ci95'][0]:+.3f}, {delta['ci95'][1]:+.3f}]")

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
