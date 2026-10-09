"""Best-of-N re-ranking with the model's own verifier.

The cheapest way to extract more useful work from a fixed model is to sample
several continuations and keep the one the model itself scores highest. No
teacher and no external weights: the score is KALIA's own mean token
log-probability for the candidate it just produced, so D2 (from-scratch-only)
stays intact and nothing is trained.

Honest limits, recorded before any number is quoted:

- The verifier is the generator. A 58M model's log-probability is a weak judge;
  the project's own research notes that search-based decoding degrades below a
  capacity threshold. This module ships the mechanism and the measurement hook,
  not a claim of improvement.
- **The scoring window is part of the mechanism, and the default window was
  measured to fail.** On the oracle-copy probe (docs/results/retrieval-probe.json,
  2026-10-08), scoring the whole 4-token continuation picked *worse* candidates
  than greedy: selected 0.547 vs greedy 0.828, delta -0.281 [-0.406, -0.172].
  A fluent wrong continuation out-scores a correct first token that continues
  awkwardly. ``score_tokens`` scores only the first K continuation tokens so the
  decision position can dominate; re-measure before trusting either setting.
- Candidates are scored teacher-forced on prompt + candidate and normalised by
  token count by default. Sum-scoring quietly favours shorter or longer
  candidates depending on the sign of the bias, so the choice is an explicit
  parameter (:func:`select_best`) rather than a default buried in the arithmetic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch

from model import GPT


@dataclass
class BestOfNResult:
    """Sampled candidates and the selection the verifier made."""

    ids: torch.Tensor
    chosen: int
    totals: list[float] = field(default_factory=list)
    means: list[float] = field(default_factory=list)
    sequences: list[torch.Tensor] = field(default_factory=list)


def _as_1d(ids: torch.Tensor) -> torch.Tensor:
    if ids.dim() == 2 and ids.size(0) == 1:
        return ids[0]
    if ids.dim() != 1:
        raise ValueError(f"expected (T,) or (1, T) ids, got shape {tuple(ids.shape)}")
    return ids


@torch.no_grad()
def continuation_logprob(
    model: GPT,
    prompt_ids: torch.Tensor,
    cont_ids: torch.Tensor,
    max_tokens: int | None = None,
) -> tuple[float, float]:
    """Total and mean log-probability of ``cont`` given ``prompt``, teacher-forced.

    ``max_tokens`` scores only the first K continuation tokens. The probe found
    that scoring the whole continuation lets fluency outvote the decision
    position, so callers that know where the decision lives should say so.

    The prompt must be non-empty: a continuation with no context has no
    predictor for its first token, and this project refuses ambiguous inputs
    rather than inventing a convention for them.
    """
    prompt = _as_1d(prompt_ids)
    cont = _as_1d(cont_ids)
    if max_tokens is not None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        cont = cont[:max_tokens]
    if prompt.numel() == 0:
        raise ValueError("prompt must be non-empty; there is no predictor for a bare continuation")
    if cont.numel() == 0:
        raise ValueError("continuation must be non-empty")
    device = next(model.parameters()).device
    seq = torch.cat([prompt, cont]).to(device).unsqueeze(0)
    logits, _ = model(seq)
    logp = torch.log_softmax(logits[0], dim=-1)
    start = prompt.numel() - 1
    targets = cont.to(device)
    gathered = logp[start : start + cont.numel()].gather(1, targets.unsqueeze(1)).squeeze(1)
    return float(gathered.sum()), float(gathered.mean())


def select_best(totals: list[float], lengths: list[int], length_normalize: bool = True) -> int:
    """Index of the highest-scoring candidate. Ties break to the earliest.

    Normalised scoring compares mean per-token log-probability; raw scoring
    compares the total. The two disagree on candidates of different lengths,
    and silently picking one is how a re-ranking result stops being comparable
    to the next one.
    """
    if not totals or len(totals) != len(lengths):
        raise ValueError("totals and lengths must be non-empty and the same size")
    if any(n <= 0 for n in lengths):
        raise ValueError("every candidate needs at least one token to be scored")
    scores = [t / n for t, n in zip(totals, lengths)] if length_normalize else list(totals)
    best = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[best]:
            best = i
    return best


def best_of_n(
    model: GPT,
    prompt_ids: torch.Tensor,
    n: int = 8,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int | None = 200,
    seed: int | None = None,
    length_normalize: bool = True,
    score_tokens: int | None = None,
) -> BestOfNResult:
    """Sample ``n`` continuations and return the verifier's pick.

    ``score_tokens`` limits scoring to the first K continuation tokens (see
    :func:`continuation_logprob`); None scores everything, which the oracle-copy
    probe measured to be worse than greedy on decision-position tasks.

    A local generator makes the whole run reproducible from ``seed``: the same
    seed draws the same candidates and therefore the same winner, which is what
    lets a later comparison attribute a difference to the model rather than to
    the dice.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    prompt = _as_1d(prompt_ids)
    if prompt.numel() == 0:
        raise ValueError("prompt must be non-empty")
    device = prompt.device
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    sequences: list[torch.Tensor] = []
    totals: list[float] = []
    means: list[float] = []
    for _ in range(n):
        out = model.generate(
            prompt.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )
        cont = out[0, prompt.numel() :]
        total, mean = continuation_logprob(model, prompt, cont, max_tokens=score_tokens)
        sequences.append(out[0])
        totals.append(total)
        means.append(mean)

    lengths = [int(seq.numel() - prompt.numel()) for seq in sequences]
    chosen = select_best(totals, lengths, length_normalize=length_normalize)
    return BestOfNResult(
        ids=sequences[chosen],
        chosen=chosen,
        totals=totals,
        means=means,
        sequences=sequences,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sum-score", action="store_true", help="rank by total log-prob instead of per-token")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    import tiktoken

    from sample import load_model

    device = torch.device(args.device)
    model = load_model(args.ckpt, device)
    encoder = tiktoken.get_encoding("gpt2")
    ids = encoder.encode_ordinary(args.prompt)
    prompt_ids = torch.tensor(ids, dtype=torch.long, device=device)
    result = best_of_n(
        model,
        prompt_ids,
        n=args.n,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        length_normalize=not args.sum_score,
    )
    print(f"candidates: {args.n} | chosen: #{result.chosen}")
    for i, (total, mean) in enumerate(zip(result.totals, result.means)):
        marker = " <-- chosen" if i == result.chosen else ""
        print(f"  #{i}: total {total:+.3f} | mean {mean:+.4f}{marker}")
    print()
    print(encoder.decode(result.ids.tolist()))


if __name__ == "__main__":
    main()
