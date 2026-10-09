"""Entity-verified decoding: keep the continuation that keeps its characters.

KALIA's documented failure mode is entity drift -- the model's own sample set
contains "A curious cat named Max" becoming "Tim" within two sentences. Best-of-N
with the model's log-probabilities cannot fix that (the oracle-copy probe
measured the log-prob selector *below* greedy), but entity retention has a
deterministic checker already in this repository: `eval_entities.py`, a regex
over capitals with a stopword list, no external models.

So: sample N continuations, score each with the rule, keep the best. The
selector is the only new thing here, and the exact configuration -- tiny model
+ rule-based entity verifier + best-of-N -- is unmeasured at this scale.

What this is not: a learned verifier, a borrowed model, a training run, or a
claim. It is the mechanism plus the measurement hook; `docs/results/` gets the
numbers before the model card gets a sentence.

Selection order is a lexicographic tuple so it is auditable and deterministic:

    (retention, mean entity span, model mean log-probability)

Retention first because it is the registered failure; span second because a
name mentioned once at the start and never again is a weaker kind of keeping;
log-probability last, as a fluency tie-break only, because it was measured to
be a poor judge on its own. Ties break to the earliest candidate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch

from best_of_n import continuation_logprob
from eval_entities import entity_report
from model import GPT


@dataclass
class EntityDecodeResult:
    """All candidates and the verifier's pick."""

    ids: torch.Tensor
    chosen: int
    texts: list[str] = field(default_factory=list)
    scores: list[dict] = field(default_factory=list)


def score_candidate(prompt: str, continuation: str, logprob: float | None = None) -> dict:
    """The verifier's reading of one candidate, via eval_entities."""
    report = entity_report(prompt, continuation)
    spans = list(report["max_span_ratio"].values())
    return {
        "retention": report["retention"],
        "mean_span": sum(spans) / len(spans) if spans else 0.0,
        "logprob": logprob if logprob is not None else float("-inf"),
    }


def select_best_index(scores: list[dict]) -> int:
    """Lexicographic (retention, mean span, logprob); ties to the earliest."""
    if not scores:
        raise ValueError("no candidate scores to select from")

    def key(i: int) -> tuple:
        row = scores[i]
        return (row["retention"], row["mean_span"], row["logprob"])

    best = 0
    for i in range(1, len(scores)):
        if key(i) > key(best):
            best = i
    return best


@torch.no_grad()
def entity_decode(
    model: GPT,
    encoder,
    prompt_text: str,
    n: int = 8,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int | None = 200,
    seed: int | None = None,
    device: torch.device | None = None,
) -> EntityDecodeResult:
    """Sample ``n`` continuations and return the entity verifier's pick.

    A local generator makes the whole run reproducible from ``seed``: the same
    seed draws the same candidates and therefore the same winner, so a later
    comparison attributes a difference to the model, not to the dice.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    device = device or next(model.parameters()).device
    prompt_ids = torch.tensor(encoder.encode_ordinary(prompt_text), dtype=torch.long, device=device)
    if prompt_ids.numel() == 0:
        raise ValueError("prompt must be non-empty")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    texts: list[str] = []
    sequences: list[torch.Tensor] = []
    scores: list[dict] = []
    for _ in range(n):
        out = model.generate(
            prompt_ids.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )
        cont = out[0, prompt_ids.numel() :]
        text = encoder.decode(cont.tolist())
        logprob = continuation_logprob(model, prompt_ids, cont)[1]
        texts.append(text)
        sequences.append(out[0])
        scores.append(score_candidate(prompt_text, text, logprob))

    chosen = select_best_index(scores)
    return EntityDecodeResult(
        ids=sequences[chosen], chosen=chosen, texts=texts, scores=scores
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time there was a cat named Max.")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    import tiktoken

    from sample import load_model

    device = torch.device(args.device)
    model = load_model(args.ckpt, device)
    encoder = tiktoken.get_encoding("gpt2")
    result = entity_decode(
        model,
        encoder,
        args.prompt,
        n=args.n,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        device=device,
    )
    print(f"candidates: {args.n} | chosen: #{result.chosen}")
    for i, row in enumerate(result.scores):
        marker = " <-- chosen" if i == result.chosen else ""
        print(
            f"  #{i}: retention {row['retention']:.2f} | span {row['mean_span']:.3f} | "
            f"logprob {row['logprob']:+.3f}{marker}"
        )
    print()
    print(args.prompt + result.texts[result.chosen])


if __name__ == "__main__":
    main()
