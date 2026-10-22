# Pre-registration — X33: does the midpoint state know whether the entity will survive?

Registered: 2026-10-22, before any GPU time. Related: X32, the entity holdout,
D57.

## Why this exists

X32 found entity attention concentrated in layer-9 heads 3, 2 and 7, but
ablating them **reduced** second-half retention by 1.17 points. The heads are
necessary, not the hiding switch. The selection failure is either downstream at
the readout, or distributed across the representation.

X33 tests the readout directly, with no second-half leakage:

- show the model the prompt plus **only the first 128 generated tokens**;
- capture the layer-9 residual state at the midpoint;
- train a linear probe on 19 stories to predict the candidate's **second-half**
  entity retention;
- test on the story it has never seen.

If a linear probe can rank candidates from the midpoint state, the information is
present and linearly decodable; the failure is at the readout. If it cannot, the
information is distributed or absent before the second half.

This is a pilot: 20 stories, 8 candidates each. A pass needs replication.

## Design

Model: `kalia-v020` at step 4,770. Prompts: the 20 frozen Gutenberg stories,
768 prompt tokens, 8 samples at temperature 0.8 / top-k 200, plus one greedy
draw — the entity-holdout protocol.

For each candidate:

- forward the prompt plus the **first 128** candidate tokens;
- capture the layer-9 block output;
- features: the mean over the first-half positions and the last-token state,
  concatenated (1,024 dimensions).

Labels: the candidate's second-half entity retention, computed after selection
and never used in the features.

Selector: ridge regression, **α = 10**, trained on the candidates from 19 stories
and tested on the held-out story; leave-one-story-out over all 20. Features are
standardized on the training stories only.

Baselines: greedy, log-probability pick, first-half entity pick, oracle, and a
shuffled-label probe.

## Pre-registered predictions

| ID | Prediction | Threshold |
|---|---|---|
| **P-1** | The midpoint state is linearly readable | probe − greedy mean second-half retention **≥ +0.010**, paired 95% CI excludes zero |
| **P-2** | The probe beats the cheap selectors | probe mean **≥** both half-pick and log-prob mean |
| **P-3** | The signal is not a label artifact | shuffled-label probe mean − greedy **< +0.010** |

P-1 is the main test. P-2 asks whether the linear readout is better than the
rules already measured. P-3 is the null: if shuffled labels do as well, the probe
is reading an artifact.

## Decision rule (fixed now)

- **P-1, P-2 and P-3 pass** → the midpoint state contains a linearly decodable
  entity signal. Replicate on a second story set before any capability claim; then
  train a selector and measure the selective-accuracy curve.
- **P-1 passes, P-3 fails** → the probe is reading an artifact; report a null.
- **P-1 fails** → no linear readout at the midpoint. Selection is nonlinear or
  distributed; close the linear-selector track and publish the null.
- **P-2 fails but P-1 passes** → the probe is real but not better than the cheap
  rules; report the readout finding without a selector claim.

## Cost

Twenty stories, 8 samples, one GPU session; generation dominates. Budget
**1.5–2 GPU-hours**. Ridge fitting and the shuffle control are CPU.

## Deviations

Any change to the layer, feature pooling, α, stories, metric or thresholds after
this commit requires a hash-registered amendment. Results are reported regardless
of outcome.
