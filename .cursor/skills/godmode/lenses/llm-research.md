# LLM research lens — models, benchmarks, training

Use for model architecture, training objectives, benchmarks, and research
claims.

## The research rule

> If the only reason to run it is that it is new, don't.
> If the only reason is a likely metric gain and it is already known, call it
> engineering, not research.

## The gate

No GPU until all seven are answered:

1. **Claim.** What is the new problem or composition, in one sentence?
2. **Closest three.** Read the abstracts, not the titles.
3. **Class.**
   - **A** new problem/composition; no exact prior match.
   - **B+** known mechanism, new setting; label it, cite the prior art.
   - **C** replication/ablation; only as a control.
   - **D** new benchmark/evaluation; do it first, usually CPU-cheap.
4. **Power.** What is the noise floor? Is the bar above ~2 standard errors?
5. **Val tax.** What is the expected language-quality cost, and what is the
   parameter-matched output-adapter control?
6. **Benchmark.** Operation-level table, not aggregate. Can a CPU probe answer
   it first?
7. **Kill.** GPU budget, 30M kill bar, and what is learned if it fails.

## Prior art

```bash
python scripts/prior_art.py "your claim" "a second phrasing"
```

Search twice, with different phrasings. Date the search. “Never tested” is a
claim with a trail, not a feeling. If a reviewer would say “this is X applied to
Y,” it is B+, not A.

## Run order

```text
problem
  -> prior art (twice)
  -> CPU probe / existing checkpoints
  -> pre-registration (hash, bars, seeds, configs)
  -> 30M screen (2 seeds)
  -> 58M only if pass
  -> 134M only if 58M passes
```

No scale without a pass. No eval tuning. No rescue.

## Red flags

- “It’s just a small module.” The val tax is not small until measured.
- “We’ll tune the weight.” That is eval tuning unless fixed before the run.
- “The aggregate improved.” Show the operation-level table.
- “One seed showed it.” Report the spread; do not promote.
- “The data baseline is close.” Then the mechanism is not the lever.
- “A bigger model will fix it.” Only if the 30M screen shows a signal.

## The audit lesson

A programme can spend most of its GPU re-testing known mechanisms under new
names. Before the next run, classify the novelty and write the kill rule. A
null that was already in the literature is a process failure, not a result.
