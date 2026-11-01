# X38 — the invariant sum works; the counterfactual objective collapses

Registered: `docs/preregistrations/2026-10-28-X38-crb-micro.md`
(sha `a28329ed…`, ledger-verified). Training kernel `kalia-x38-crb`
(code `1c36390`, **5.16 GPU-h**), evaluation kernel `kalia-x38-eval`
(code `5f85b8c`). Eight arms, 30M, 763 steps, seeds 1338/1339.

**Verdict: `advance = false`.** C-1, C-1b, C-4 and C-6 pass; **C-2, C-3 and
C-5 fail**. The invariant-sum decomposition does not tax language quality, but
the counterfactual loss collapses the stable stream to **rank ~1.5** and does
not improve the real swap probe. No 58M scale.

## The mechanism

Each block's residual delta is split by a shared orthonormal rank-64
projection:

```text
stable_l   = U D delta_l
volatile_l = delta_l - stable_l
s = s + stable_l
v = v + volatile_l
x = s + v
```

The sum is the standard residual, so the LM logits are invariant to the split
by construction. Only the paired counterfactual loss trains `U`/`D`: invariant
pairs (consistent rename, synonym) are pulled together; sensitive pairs (role
swap, negation, number, spatial change) are pushed apart.

## The result

| arm | val s1338 | val s1339 | swap s1338 | swap s1339 | paraphrase s1338 | paraphrase s1339 |
|---|---:|---:|---:|---:|---:|---:|
| control | **4.1468** | **4.1594** | −0.00052 | +0.00062 | 0.14604 | **0.13520** |
| split | 4.1474 | 4.1638 | **+0.00298** | **+0.00293** | 0.14950 | 0.14708 |
| loss | 4.1583 | 4.1528 | +0.00123 | +0.00088 | **0.14433** | **0.11119** |
| **CRB** | 4.1493 | 4.1613 | +0.00236 | +0.00231 | 0.17218 | 0.14450 |

Paraphrase is a normalised distance; **lower is better**. Swap is
`NLL(swapped) − NLL(original)`; a role-sensitive model should be more
surprised, so higher is better.

| bar | measured | verdict |
|---|---|---|
| **C-1** CRB val ≤ split + 0.005 | **+0.0019 / −0.0025** | **PASS** |
| **C-1b** split val ≤ control + 0.005 | **+0.0006 / +0.0044** | **PASS** |
| **C-2** CRB swap ≥ split + max(spread, 0.01) | **−0.00062 / −0.00062** | **FAIL** |
| **C-3** CRB paraphrase ≤ split − 0.01 | **−0.02268 / +0.00258** (split − CRB) | **FAIL** |
| **C-4** CRB retention ≥ split − 0.005 | **−0.00069 / 0.0** | PASS (floor-limited) |
| **C-5** stable rank > 8, norm > 0.01, change > 1e-3 | rank **1.48 / 1.44**; norm **1319 / 1073**; change **1.49–1.52** | **FAIL on rank** |
| **C-6** pair audit clean | templates/names disjoint; hashes match | PASS |

Retention is **0.0–0.0007 for every arm**: the greedy probe is floor-limited at
this scale, so C-4 passes trivially and is not evidence of retention.

## What happened

**The invariant sum is free at the language level.** Split and control are the
same model on val (C-1b), and the CRB does not pay a language tax (C-1). That is
the one real positive: the decomposition is a reparameterisation, and the LM
path does not notice it.

**The objective collapsed the stable stream.** The CF loss did train the
projection — the relative change from initialisation is **1.49–1.52** — but the
stable stream's effective rank is **1.48 / 1.44**, not the expected 8+. A
single direction that separates the invariant pair set from the sensitive pair
set is enough to minimise the pairwise loss. The stable stream became an **edit
detector**, not a meaning representation. Its norm grew to **1073–1319** against
the split's **72**, while the fixed RMS norm in the loss removed any penalty for
scaling that one direction.

**The split did not help, and may have hurt.** The CRB is slightly **less**
swap-sensitive than the split on both seeds (−0.00062), and its paraphrase
effect is inconsistent: worse on 1338 by 0.0227, slightly better on 1339 by
0.0026. The **loss-only** arm is the interesting control: it reduces paraphrase
distance on both seeds (0.1443 / 0.1112 vs split 0.1495 / 0.1471) but also
reduces swap sensitivity (0.00123 / 0.00088 vs split 0.00298 / 0.00293). The
counterfactual objective as designed teaches **invariance**, not sensitivity, on
the real story probe. The split is not the lever.

## What this establishes

- **The invariant-sum residual decomposition is a valid free reparameterisation**
  at 30M: C-1b and C-1 pass. It is a component candidate, not a model win.
- **The paired counterfactual objective, as implemented, fails.** It trains the
  projection but collapses it to rank ~1.5 and does not transfer to the real
  swap probe.
- **The rank bar caught a real failure mode.** Without C-5, the trained
  projection and the language-neutral val would have looked like a partial win.
- **The loss-only arm is a lead, not a result.** It improves paraphrase
  invariance on both seeds while reducing swap sensitivity; a new registration
  would need a sensitivity/reconstruction guard before calling that a mechanism.

## What this does not establish

- Not a 58M result; the 30M micro is a screen.
- Not a claim that all counterfactual residual decompositions fail; this
  objective, pooling and pair design do.
- Not a retention result: the greedy probe is floor-limited.
- Not a paper claim. A negative/analysis paper is possible, but the positive
  mechanism claim is not supported.

## Next, if this line is reopened

A **new registration**, not a tune of this run:

1. **Rank/coverage regulariser or reconstruction head** on the stable stream, so
   a rank-1 edit detector cannot satisfy the loss.
2. **Sensitivity-only or balanced objective**; the rename-invariance term is the
   likely cause of the reduced real swap sensitivity.
3. **Story-level pairs** (entity roles in real continuations), not templates.
4. Only if a micro clears C-1/C-2/C-3/C-5 then consider 58M.

The portfolio's next bets are **Relational Algebra Attention** (the most novel)
and **Predictive-Coding Depth** (the most practical); CRB as implemented is
closed.

## Artifacts

- `x38_summary.json` — vals, swap, paraphrase, retention, stable, C-1..C-6.
- `x38_eval_manifest.json` — checkpoint/config/pair hashes, eval code.
- `run_manifest-train.json` — the training run's configs, params and pair hashes.
- `configs/x38-*.yaml` — the eight arm configs.
- `val_logs/`, `train_logs/` — the eight training curves.
- `pairs_audit.json` — the counterfactual pair audit (hash matches the run).
- `kalia-x38-crb-train.log` — the training kernel log (includes the eval bug).
- `kalia-x38-eval.log` — the eval-only kernel log.
