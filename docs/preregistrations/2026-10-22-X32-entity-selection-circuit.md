# Pre-registration — X32: the entity-selection circuit

Registered: 2026-10-22, before any GPU time. Related: entity-holdout (2026-10-21),
D55, D56, the 2026-10-22 novelty check.

## Why this exists

The entity holdout showed that the model's candidate pool contains a better
continuation than greedy — the oracle reaches **0.0618** on the unseen half
against greedy's **0.0170** — while every cheap selector fails. The model is not
ignorant. Something in the output path is hiding the answer.

The next question is mechanistic, not behavioural: **is there a specific
attention head (or a small set of heads) that carries the wrong-name signal, and
does ablating it improve the continuation?**

This is a pilot. Twenty stories is enough to find a large circuit, not enough to
certify a small one. A pass needs replication.

## Design

Model: `kalia-v020` at step 4,770. Prompts: the 20 frozen Gutenberg stories in
`eval/ttt_probe_stories.json`, prompt 768 tokens, 256 generated, 8 samples at
temperature 0.8 / top-k 200, plus one greedy draw — the same protocol as the
entity holdout.

1. Generate the candidate pools with the unmodified model.
2. For every prompt + candidate, run a forward pass with a manual attention path
   that records, for each layer and head, the mean attention mass from the
   continuation positions to the prompt's entity tokens.
3. Rank heads by **entity enrichment**: entity attention mass divided by the mass
   a uniform attention row would give those positions.
4. Ablate the top three heads by zeroing their value contribution at inference.
   Generate one greedy continuation per story with those heads off, and score
   **second-half entity retention** — the same held-out-half metric as the entity
   holdout.
5. Repeat with three random, non-top heads as a specificity control.
6. Measure the model's NLL on the 20 reference continuations with and without the
   ablation, so a retention change cannot be bought by destroying the model.

## Pre-registered predictions

| ID | Prediction | Threshold |
|---|---|---|
| **H1** | Entity information is concentrated in identifiable heads | at least one head with entity enrichment **≥ 3×** uniform, and the top three all **≥ 2×** |
| **H2** | Ablating the top three heads improves held-out retention | mean second-half retention − greedy **≥ +0.010**, paired 95% CI excludes zero |
| **H3** | The effect is specific to those heads | random-three mean delta **< +0.010** |
| **H4** | The ablation does not destroy the model | reference NLL increase **≤ 5%** relative |

H2 is the main test. H1 is the mechanism check: if no head carries entity mass,
there is no circuit to ablate and H2 is unreadable. H3 controls for generic
degradation; H4 controls for the obvious failure mode where a broken model emits
the prompt's names by accident.

## Decision rule (fixed now)

- **H1, H2, H3 and H4 pass** → a causal entity-selection component is identified
  at 58M. Replicate on a second seed / larger story set before any capability
  claim.
- **H2 passes, H3 or H4 fails** → the gain is not attributable to the named heads,
  or it is bought by model damage. Report as a null.
- **H2 fails** → no single-head intervention at this scale. Report the head map
  and the null; do not keep ablating heads until one works.
- **H1 fails** → no concentrated entity head. The selection failure is distributed;
  report that as the result.

## Cost

Twenty stories, 8 samples, one GPU session. Generation is the dominant cost; the
attention passes and ablations are short. Budget **1.5–2 GPU-hours**.

## Deviations

Any change to the heads, stories, metric, thresholds or ablation after this commit
requires a hash-registered amendment. Results are reported regardless of outcome.

---

## Result (appended 2026-10-22, after the run)

Kernel `subhajitlucky/kalia-x32-entity-circuit`, COMPLETE, one GPU session.
Twenty stories, 180 prompt + candidate passes, 20 greedy ablations.

**H1 PASS.** Entity attention is concentrated in layer 9: heads 3, 2 and 7 have
enrichment **3.45×, 2.37× and 2.12×** uniform; no other head clears 2×.

| arm | mean second-half retention | Δ vs greedy | 95% CI |
|---|---|---|---|
| top three heads ablated | — | **−0.0117** | [−0.0263, +0.0000] |
| random three heads ablated | — | −0.0052 | [−0.0291, +0.0199] |

Top minus random: −0.0065 [−0.0284, +0.0124]. Reference NLL: 3.7182 → 3.7574,
**+1.05%**.

- **H2 FAIL** — the top-three ablation lowers retention; they are necessary, not
  the hiding switch.
- **H3 PASS** — the random control does not improve by the H2 bar.
- **H4 PASS** — the ablation does not destroy the model.

**Decision: no single-head intervention. Report the null.** The next test is a
linear probe or activation steering on layer-9 residual states; more head
ablation is not justified by this result.

Artifacts: `docs/results/x32/`.
