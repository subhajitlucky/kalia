# Pre-registration — X37: does a Pāṇinian rule grammar beat a matched dense MLP?

Registered: 2026-10-27, before any GPU time on X37. Related: D48, D60, X36,
and the ancient-text research in `docs/research/2026-09-23-ancient-texts-algorithms.md`.

## Why this exists

The post-v0.2.0 experiments kept adding **memory** or **selection** to an
unchanged transformer. X37 tests a different unit of computation: a small,
self-selected **grammar of reusable rules**, mapped from Pāṇini's Aṣṭādhyāyī.

The mapping is mechanical, not metaphorical:

| Aṣṭādhyāyī | SUTRA mechanism |
|---|---|
| **Sūtra** | a reusable low-rank rule in a learned codebook |
| **Anuvṛtti** | a carry state from the previous chunk's rule usage |
| **Adhikāra** | the previous chunk's mean as a scope vector |
| **Utsarga** | an always-on general SwiGLU path |
| **Apavāda** | a sparse exception path opened by a conflict detector |
| **Pratyāhāra** | (reserved for the next stage: hierarchical rule clusters) |

The question: does a small set of shared rules — selected per token, carried
across a chunk, scoped per chunk — beat a **parameter-matched dense MLP** on
language modelling and on a compositional swap test?

## The architecture

Each transformer block's MLP is replaced by a SUTRA block:

```text
general(x)  +  rule_{top-1}(x; router(x, carry, scope))  +  exception(x; gate)
```

- **Rules:** 64 low-rank transformations, rank 8, shared across tokens in a
  block. Top-1 selection.
- **Router:** sees the token, the anuvṛtti carry (64-d), and the adhikāra scope
  (the previous chunk's mean).
- **Carry:** updated at a chunk end from the chunk's mean rule distribution and
  a learned carry gate; read only by the next chunk.
- **Scope:** the previous chunk's mean; read only by the next chunk.
- **Balance:** Switch-style load-balancing loss (weight 0.01) so the codebook
  does not collapse.
- **Exception:** optional; a rank-4 rule with a conflict gate initialised closed
  (-4) and a sparsity penalty.

The carry/scope are updated **after** a chunk and read only by the **next**
chunk, so the grammar cannot see the future.

## Arms

30M micro, `micro-muon` + QK-Norm, context 512, 763 steps, seeds **1338/1339**.

| arm | what it is | params |
|---|---|---:|
| `base` | standard micro MLP | 29,921,280 |
| `control` | **wider SwiGLU, parameter-matched to SUTRA** | 32,575,488 |
| `rules` | SUTRA rules + router; no carry/scope | 32,627,718 |
| `full` | rules + anuvṛtti carry + adhikāra scope | 32,627,718 |
| `exception` | full + utsarga/apavāda | 32,651,148 |

The control/full difference is **0.16%**. The base is the standard-model anchor;
the control is the capacity control. 10 runs total, ~5–6 GPU-h.

## Metrics

1. **Canonical val loss** (the micro val, same as every micro arm).
2. **Swap sensitivity** — a compositional test:
   - 20 frozen stories; first 384 prompt tokens; next 128 reference-continuation
     tokens.
   - Find two capitalized names in the prompt; swap them; tokenize the swapped
     prompt; keep the **same continuation**.
   - `sensitivity = NLL(swapped prompt, continuation) − NLL(original prompt,
     continuation)`, mean over stories with ≥2 names.
   - A role-sensitive model should be more surprised by the original
     continuation after the swap.
3. **Rule usage** — mean hard usage across the 6 SUTRA blocks on a val batch:
   entropy, max share, effective rules.
4. **Exception rate** — mean conflict-gate value (exception arm).
5. **Entity retention** — the X36 short-context protocol, as a secondary.

## Bars

| ID | Prediction | Threshold |
|---|---|---|
| **S-1** | The grammar does not cost language quality | `full val − control val ≤ +0.005` |
| **S-2** | The grammar composes roles | `full swap − control swap ≥ max(control seed spread, +0.02 nats)`, both seeds the same direction |
| **S-3** | The codebook stays alive | usage entropy > **2.0 bits**, max rule share < **0.5**, effective rules > **8**, both seeds |
| **S-4** | The ruler is reported | control seed spread on val and swap |
| **S-5** | The carry/scope is not decorative | `full` vs `rules` reported; if `full` is worse on both val and swap, the carry/scope is not helping |

## Decision rule (fixed now)

- **S-1, S-2 and S-3 pass** → the rule grammar is a candidate. Register the 58M
  screen; no capability claim from the micro.
- **S-1 passes, S-2 fails** → no compositional gain; the grammar is not doing
  the job it exists for. Close this architecture, keep the usage result.
- **S-1 fails** → the rules cost language quality at this scale; no scale.
- **S-3 fails** → router collapse; fix the routing/load balance before reading
  S-2, not the eval.
- **S-5 shows `rules` ≈ `full`** → the Paninian carry/scope is not the lever;
  report the codebook-only result and do not scale the full version.
- **Exception**: if it improves val/swap without reducing usage entropy, keep
  it as a component; otherwise drop it.

## What this does not establish

- Not a 58M result; the micro is a screen.
- Not a world-first: MoE, Switch Transformers, LoRA/adapter banks, hypernetworks
  and Panini-inspired NLP all exist. The novel combination is a **self-selected,
  carried, scoped low-rank rule grammar with a general/exception path**,
  evaluated on compositional swaps and entity retention at sub-100M.
- Not a claim that Sanskrit grammar is a better inductive bias in general; it is
  a testable mapping, and the micro decides.

## Cost and stopping

10 arms × 763 steps ≈ **5–6 GPU-h**, plus the swap/usage eval. If a slot is not
free until the next quota window, the run waits; the registration does not
expire. If the router collapses (S-3 fails), stop and fix the routing; do not
tune the eval.

## Artifacts (to be archived)

- `docs/results/x37/` — summaries, configs, val logs, rule usage, swap rows,
  run manifests.
- `tools/kalia_x37_sutra_kernel.py`.
- `kaggle/kernel-metadata-x37-sutra.json`.

---

## Amendment 1 (2026-10-27, after X37 v1, before the rerun)

X37 v1 completed **base and control**, then failed all six rule arms at step 0:
Muon's split treated every parameter with `ndim >= 2` as a hidden matrix and
sent the 3D rule codebooks to Newton–Schulz, which asserts 2D. The split is now
`ndim == 2`; 3D rule tensors go to AdamW. Base/control have no 3D parameters,
so their v1 values are valid and are reused from the archived v1 log; the six
rule arms are rerun with the fix. No arm, seed, threshold or decision rule
changes. The v1 log is archived at `docs/results/x37/kalia-x37-sutra-v1.log`.

---

## Result (appended 2026-10-28, after the registered runs)

Kernels `kalia-x37-sutra` v1 and v2; v1 completed base/control then died on the
3D-Newton–Schulz optimizer bug; v2 reran the six rule arms with the `ndim == 2`
fix and reused the valid base/control. ~14 GPU-h across both launches.

| arm | val s1338 | val s1339 | mean |
|---|---:|---:|---:|
| control (matched MLP) | **4.0692** | **4.2290** | **4.1491** |
| rules | 4.1503 | 4.2113 | 4.1808 |
| **full** | 4.1721 | 4.2564 | **4.2143** |
| full + exception | 4.1932 | 4.3872 | 4.2902 |

- **S-1 FAIL** — full − control **+0.0652**.
- **S-2 FAIL** — swap deltas **−0.00128 / −0.00059**; the full is less
  swap-sensitive than the control.
- **S-3 PASS** — entropy 4.157, max share 0.018, effective **63.9 / 64**.
- **S-4** — control val spread 0.160; swap spread 0.0026; reported.
- **S-5** — full − rules val **+0.0335**; full − rules swap +0.0026/+0.0020;
  the carry/scope is not helping val.

**Registered verdict: `advance_full = false`. No 58M scale.** The router and
codebook work, but uniformly: all 64 rules are used at ~1.6%, so the codebook is
acting as capacity, not as a specialized grammar. The dense control is better on
val and at least as good on the swap. A new registration with a targeted change
(rule specialization/diversity, a matched general path, a longer micro) is
required before this line is reopened. Artifacts: `docs/results/x37/`;
decision: D62.
