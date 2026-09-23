# Amendment 1 — v0.2.0 Promotion Rules (documentation correction)

References: `docs/preregistrations/2026-09-23-v020-promotion-rules.md`
(registered hash `f8aa49291619bf0bbb7e9a7f3cceb9ec34884675926660ab5cda552ec88264be`)

## What is corrected

The registered document describes the recipe base as "Muon+ col_row at LR 0.02".
That description is factually wrong. `muon_plus` was never present in
`configs/kalia-m.yaml` (verified across all commits; the v0.1.2 commit message
reads "lock winning recipe - Muon + QK-Norm + softcap"). **v0.1.2 was trained
with plain Muon at LR 0.02.** Muon+ was validated only at micro scale (E1:
3.4941 vs 3.5091) and fell below the 0.02-nat promotion threshold (D15), so it
was correctly not promoted to the full-scale recipe.

## Corrected base

The v0.2.0 recipe base is the frozen v0.1.2 recipe: **plain Muon at LR 0.02 +
QK-Norm + logit soft-cap 30 + cosine schedule**. Muon+ remains excluded
(below threshold); promoting it would require a new micro-ablation at a larger
token budget.

## Why this is not a result-driven change

The corrected fact was already true at registration time; this amendment fixes
a description error, not a decision. No promotion rule, metric, threshold, or
decision criterion changes. All other terms of the registered document stand.
