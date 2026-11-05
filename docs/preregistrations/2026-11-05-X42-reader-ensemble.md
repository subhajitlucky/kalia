# Pre-registration — X42: Reader-Guided Compiler Ensemble (RGCE)

Registered: 2026-11-05, before the X42 selection result. CPU only.

## Why this exists

X41 trained two edit-compiler adapters (Arm B, seeds 1401/1402) and evaluated
them end-to-end. The compilers learn single edits (0.993–0.997 exact match) and
recover `number+place` at the gold-edit ceiling (**0.694** on both seeds). The
two hard-cell results are:

| seed | S-P | `number+place` | `rename+swap` |
|---|---:|---:|---:|
| B_1401 | 0.4275 | 0.6935 | **0.3537** |
| B_1402 | 0.4100 | 0.6935 | **0.4830** |

The registered X41 rule requires `rename+swap` ≥ 0.45 on **both** seeds, so X41
is a partial advancement, not a clean pass. The two adapters make different
`rename+swap` errors. X42 asks whether the **frozen reader** can select between
their edits without labels, using only the candidate likelihoods it already
computes.

## Mechanism

For each record, each compiler seed produces an edited prefix. The frozen
v0.3.1 reader scores every candidate under each prefix. The selector chooses
one edit from those distributions and answers with that edit's argmin-NLL
candidate. No labels, no retraining, no new generation.

### Pre-registered selection rules

1. **MaxProb** — highest maximum candidate probability (softmax over −NLL).
2. **Margin** — highest top-1 minus top-2 probability.
3. **MinEntropy** — lowest candidate-distribution entropy.
4. **Agreement** — if both edits predict the same candidate, use it; otherwise
   MaxProb.
5. **MaxProbNoNoop** — MaxProb among edits whose prefix differs from the
   factual prefix; fall back to MaxProb if both are no-ops.
6. **Oracle** — choose the edit that yields the correct candidate. Upper bound
   only; not deployable.

Ties prefer the B_1402 edit.

## Split

Within each of the four cells, records are assigned to **dev** or **test** by a
deterministic hash of the record id (`sha256(id + "x42") % 2`), 50/50. The
selection rule is chosen on **dev** by S-P accuracy (tie: `rename+swap`), then
evaluated **once** on **test**. The full-set result is reported as secondary.

## Bars (held-out test half)

| ID | prediction | threshold |
|---|---|---|
| **T-1** | `number+place` survives selection | ≥ 0.40 |
| **T-2** | `rename+swap` is repaired | ≥ 0.45 |
| **T-3** | S-P aggregate | ≥ 0.40 |
| **T-4** | U-P aggregate | ≥ 0.30 |
| **T-5** | no label leakage | selection uses only NLLs/candidates |
| **T-6** | selection is deterministic | fixed split and tie rule |

Also report the best single seed on the same test half and the Oracle upper
bound. The primary comparison is the selected rule versus the best single seed.

## Decision rule

- **T-1…T-4 pass on test:** the reader-guided ensemble is a specific
  advancement; record a GitHub release candidate with the selector and both
  adapters.
- **T-2 fails:** the ensemble does not repair the unstable cell; log the null.
  No tuning of the selection rule after the test result. A different rule set
  is a new registration.
- The public base model remains v0.3.1; this is a system-level experiment.

## Cost

CPU only: reads the saved per-record NLLs from the X41 eval. No GPU.

## Artifacts

- `docs/results/x42-reader-ensemble/` — split, dev rule table, test table,
  per-record selections, script output.
- `tools/x42_reader_ensemble.py`.
- This file.
