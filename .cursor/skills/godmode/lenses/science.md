# Science lens — biology, experiments, empirical research

Use for hypotheses, experiments, diagnoses, literature reviews, and causal
claims.

## Frame

- What is the claim, in one falsifiable sentence?
- What is the mechanism? If there is no mechanism, say so.
- What would prove the claim wrong? If nothing would, it is not a claim.
- What is the prior art? Search twice with different phrasings.

## Prior art

Run:

```bash
python scripts/prior_art.py "your claim" "a second phrasing"
```

Read the closest three abstracts. For each, write one line: *why this is not
the same experiment.* If the closest paper already tests the same mechanism,
data, or split, it is a replication, not a discovery.

## Design

- What is the control? Is it matched on the thing that matters?
- What are the confounders? Name the three most likely.
- What is the intervention, exactly? What is the dose, duration, and timing?
- What is the outcome, and is it measured blind?
- What is the sample size, and what effect can it detect? Compute power before
  the run.
- Is the result robust to a different measure, seed, batch, or model?

## Evidence

- Distinguish correlation, mechanism, and causation.
- Report effect size and uncertainty, not only “significant.”
- Check for p-hacking, selective reporting, and publication bias.
- Prefer replication over a single striking result.
- Date every number and name its source.

## Decision

- What is the cheapest experiment that would change your mind?
- What is the cost of a false positive vs a false negative?
- If the result is null, what did you learn? If the answer is “nothing,” do not
  run it.
- Pre-register the bar, the sample, and the kill rule when the result matters.

## Output

Answer first. Then the mechanism, the evidence, the confounders, the
uncertainty, and the next experiment. State what would falsify the conclusion.
