# X37 — the Pāṇinian rule grammar does not beat a matched dense MLP

Registered: `docs/preregistrations/2026-10-27-X37-sutra-micro.md`
(sha `108ad5fe…`, ledger-verified). Kernels `kalia-x37-sutra` v1 and v2
(`kalia-x37-sutra` version 2, code `983d706`), **~14 GPU-h** across both
launches. v1 completed base/control, then died on a 3D-Newton–Schulz optimizer
bug; v2 reran the six rule arms with the fix and reused the valid v1
base/control. The v1 log is `kalia-x37-sutra-v1.log`.

## The idea

Replace each block's MLP with a small grammar of reusable low-rank rules:
64 rules, top-1 routing, an **anuvṛtti** carry, an **adhikāra** scope, an
always-on general SwiGLU, and an optional **utsarga/apavāda** exception path.
The control is a **parameter-matched wider SwiGLU** (32,575,488 vs 32,627,718
params, 0.16%). The micro is 6×384×6, 763 steps, seeds 1338/1339.

## Result

| arm | val s1338 | val s1339 | mean |
|---|---:|---:|---:|
| control (matched MLP) | **4.0692** | **4.2290** | **4.1491** |
| rules | 4.1503 | 4.2113 | 4.1808 |
| **full** (rules + carry + scope) | 4.1721 | 4.2564 | **4.2143** |
| full + exception | 4.1932 | 4.3872 | 4.2902 |

| bar | measured | verdict |
|---|---|---|
| **S-1** full val ≤ control + 0.005 | **+0.0652** | **FAIL** |
| **S-2** full swap − control ≥ max(spread, +0.02), both seeds | **−0.00128 / −0.00059** | **FAIL** |
| **S-3** usage entropy > 2.0, max share < 0.5, effective > 8 | **4.157**, **0.018**, **63.9 / 64** | **PASS** |
| **S-4** ruler reported | control val spread 0.160; swap spread 0.0026 | reported |
| **S-5** carry/scope not decorative | full − rules val **+0.0335**; swap +0.0026 / +0.0020 | **not helping val** |

Swap sensitivity: control 0.00434 / 0.00179; full 0.00306 / 0.00120; rules
0.00044 / −0.00080. The full model is **less** sensitive to the name swap than
the control, not more.

**`advance_full = false`.** The rule grammar as implemented does not beat the
parameter-matched dense MLP on language modelling or on the compositional swap
test. No 58M scale.

## The one positive, and what it says

**S-3 passes decisively:** all 64 rules are used, entropy 4.157 of 4.159,
max rule share 1.8%, effective rules **63.9 / 64**. The router and the
load-balancing loss work; the codebook does not collapse.

That is also the clue. A uniform router is not a **specialized** grammar: every
rule is used at roughly the same rate, so the codebook is acting like extra
capacity, not like 64 distinct rules. The full model is worse than the control
on val, and the carry/scope is worse than the rules alone on val (+0.0335). The
rules are not earning their keep.

## What this establishes

- **The SUTRA rule grammar, as implemented, is a null at 30M/763 steps.**
  The control's dense capacity is better on val; the grammar is not better on
  the compositional swap.
- **The codebook/router mechanism itself works** (S-3); the failure is not a
  training crash or a dead codebook.
- **A uniform router is a design signal.** Specialization, not capacity, is
  what the next iteration must produce.
- **The Paninian mapping is still an untested idea at scale.** One 30M/763-step
  implementation failing does not prove the grammar cannot work; it proves this
  version does not.

## What this does not establish

- Not a 58M result; the micro is a screen.
- Not evidence that all rule-grammar designs fail; the next version must change
  a specific mechanism (rule specialization/diversity, a larger general path,
  longer training, top-k routing, role-conditioned routing, or scale).
- Not a capability claim in either direction.

## Next, if this line is reopened

A **new registration** with one targeted change, not a tune of the same run:

1. **Specialize the rules**: a diversity/entropy objective on rule assignments,
   or route on a token/role signal, so the 64 rules are not interchangeable.
2. **Fair general path**: the SUTRA general is 1,024 wide against the control's
   1,408; match it before attributing the val gap to the rules.
3. **Longer micro**: 763 steps may be too short for a sparse rule codebook; a
   1,500-step screen is the next budget.
4. Only if the micro clears S-1/S-2 then consider 58M.

## Artifacts

- `x37_summary.json` — vals, swap, usage, S1/S2/S3/S5, verdict.
- `run_manifest-v2.json` — the six rule-arm configs and hashes; the reused v1
  base/control.
- `configs/x37-*.yaml` — the six v2 configs.
- `val_logs/x37-*.csv` — the six v2 val curves.
- `kalia-x37-sutra-v1.log` — the v1 optimizer-bug launch (base/control valid).
- `kalia-x37-sutra-v2.log` — the six rule-arm launch.
