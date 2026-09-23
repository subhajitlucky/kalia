# Pre-registration — X16 Chakravyuha Reversal Training

Registered: 2026-09-23 (before any X16 run starts)
Related: decision D37; `docs/research/2026-09-23-chakravyuha-mechanism.md`

## Hypothesis

- **H1**: Training on sequences where 50% are randomly segment-reversed (chunk
  lengths 4–16 tokens) reduces the **Abhimanyu gap** (reverse-token NLL minus
  forward NLL) by **≥ 0.10 nats** versus control at equal total tokens, without
  increasing forward validation loss by more than 0.02.

## Conditions

| Arm | Config | Data |
|---|---|---|
| control | `micro-muonplus-qk` | standard order |
| treatment | identical config | 50% random-segment reversal |

Both arms: 500 steps, same seed list, same token budget per step.

## Metrics

- Primary: `abhimanyu_gap` from `eval_reversibility.py` on
  `eval/probe_sentences.json`
- Secondary: validation loss (and bpB), forward NLL

## Analysis plan

- Seeds 1337 and 1338 per condition; both reported; verdict on the seed-pair mean.
- Fixed 500 steps; no early peeking; results reported regardless of outcome.

## Predictions

- **P1**: control gap > 0.3 nats (the model can enter but not exit).
- **P2**: treatment gap < control gap − 0.10 nats.
- **P3**: forward validation-loss change within ±0.02 (no harm in the forward
  direction).

## Deviations policy

Any change to this plan after runs begin is recorded as a deviated result with
the reason stated.
