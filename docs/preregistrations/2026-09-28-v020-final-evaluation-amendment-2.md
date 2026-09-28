# Amendment 2 — v0.2.0 Final Evaluation (documentation correction)

References: `docs/preregistrations/2026-09-28-v020-final-evaluation.md`
(registered hash `7f180dfa982473b02e81d56775e6a8f47d29d0849722886edc36b17f26cb8a08`)

## What is corrected

Forbidden claim 4 describes the in-training validation curve as "a 20-batch
in-training estimate". The v0.2.0 config sets `eval_steps: 50`, so the training
loop evaluates **50 batches**, not 20. (`configs/kalia-v020.yaml` line 27; the
figure generator has said "50 batches each" all along, which is why the error sat
in the registered text rather than in the published chart.)

## Corrected statement

Forbidden claim 4 reads: the training-loop val curve is a **50-batch**
in-training estimate, noisy by design; the headline is the deterministic
**100-batch** number from `eval_val.py`.

## Why this is not a result-driven change

The error was present at registration time and was found while regenerating the
figures, with no v0.2.0 result in hand. No threshold, metric, protocol, yardstick
or claim boundary changes — the substance of the claim (the training curve is not
the headline, and is published as context only) is untouched. As with Amendment 1
to the promotion rules, a factual description error in a registered document is
corrected by hash-registered amendment rather than by editing the registered
file, so the original hash still verifies.
