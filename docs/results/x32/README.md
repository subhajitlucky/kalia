# X32 — the entity-selection circuit: the heads are necessary, not the switch

Registered: `docs/preregistrations/2026-10-22-X32-entity-selection-circuit.md`
(sha `c3a6a2f2…`, ledger-verified). Kernel
`subhajitlucky/kalia-x32-entity-circuit`, code `8843297`, one GPU session.

The entity holdout showed the candidate pool contains a better continuation than
greedy — oracle **0.0618** on the unseen half against greedy's **0.0170** — while
every cheap selector fails. X32 asked whether a specific attention head carries
the wrong-name signal, and whether ablating it improves the continuation.

## Design

20 frozen stories, 768 prompt tokens, 256 generated, 8 samples at temperature
0.8 / top-k 200, plus one greedy draw. The pools were generated with the
unmodified model. A manual attention path then recorded, for every prompt +
candidate, the attention mass from continuation positions to the prompt's entity
tokens. The top three heads by **entity enrichment** were ablated, along with a
random-three control; second-half retention was scored against greedy, with a
reference-text NLL guard.

## Head map

The entity mass is concentrated in **layer 9**. The top three heads are:

| layer | head | entity mass | uniform | enrichment | recent mass | entropy |
|---|---|---|---|---|---|---|
| 9 | 3 | 0.01201 | 0.00349 | **3.45×** | 0.00713 | 0.413 |
| 9 | 2 | 0.00826 | 0.00349 | **2.37×** | 0.00654 | 0.485 |
| 9 | 7 | 0.00741 | 0.00349 | **2.12×** | 0.00657 | 0.425 |

These are the only heads above the pre-registered 2× bar; the next heads are
below it. The enrichment is real: the same heads also have elevated
recent-token mass, so they are not simply attending to every prompt position.

## Ablation

| arm | mean second-half retention | Δ vs greedy | 95% CI |
|---|---|---|---|
| greedy | — | — | — |
| **top three heads ablated** | — | **−0.0117** | [−0.0263, +0.0000] |
| random three heads ablated | — | −0.0052 | [−0.0291, +0.0199] |

Top minus random: **−0.0065** [−0.0284, +0.0124].

Reference NLL: **3.7182 → 3.7574**, **+1.05%**. The ablation does not destroy the
model.

## Registered bars

| ID | bar | measured | verdict |
|---|---|---|---|
| **H1** | one head ≥3× uniform; top three ≥2× | 3.45×; 2.37×, 2.12× | PASS |
| **H2** | top-three retention − greedy ≥ +0.010, CI excludes zero | **−0.0117**, CI includes zero | **FAIL** |
| **H3** | random-three mean < +0.010 | −0.0052 | PASS |
| **H4** | reference NLL increase ≤5% | +1.05% | PASS |

**Registered verdict: no single-head intervention. Report the null.**

## What this means

**The entity information is in layer 9, but the heads are necessary, not
harmful.** Ablating the three most entity-attentive heads *reduces* retention by
1.2 points. They are part of how the model keeps names, not the thing hiding the
right one. The hypothesis that one wrong-name head could be switched off is not
supported.

The selection failure is therefore not a simple “bad head” to remove. It is
either:

- a **readout/decision** problem downstream of layer 9, where the representation
  is present but the next-token distribution still prefers the wrong name; or
- a **distributed** competition across several heads, none of which is the
  switch.

The random control is also slightly negative, so this pilot cannot resolve
whether the top heads are *specifically* more necessary than random ones — the
top-minus-random interval includes zero. H1 is the solid result: entity mass is
concentrated at layer 9.

## Caveats

Twenty stories, one seed, greedy ablation. This is a pilot: it can find a large
circuit, not certify a small one. The next test should be a linear probe or
activation steering on layer-9 residual states, not more head ablation.

## Artifacts

- `x32_summary.json` — verdict, per-story rows, top heads.
- `x32_heads.json` — the full 10×8 head map.
- `run_manifest.json` — code fingerprint, checkpoint hash, story hash, bars.
- `kalia-x32-entity-circuit.log` — the kernel log.

Reproduce the verdict from the registered JSON; the decision rule is in the
pre-registration.
**Decision:** recorded as D57 — no further head ablation; next is a linear probe or steering on layer-9 states.
