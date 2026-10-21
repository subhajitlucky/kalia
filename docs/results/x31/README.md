# X31 — QK-Norm earns its place; the logit soft-cap does not

Registered: `docs/preregistrations/2026-10-21-X31-qk-softcap-isolation.md`
(sha `5080eaf7…`, ledger-verified). Kernel
`subhajitlucky/kalia-x31-qk-isolation`, code `efe15c0`, one machine, 3.8 GPU-h.

D8 adopted QK-Norm + logit soft-cap on one 30M run at seed 1337: Muon 3.5937 →
Muon + both **3.5103**, a further **−0.0834**. It was never replicated and the
two components were never isolated. X31 ran the 2×2 at seeds 1338/1339:

| condition | qk_norm | soft-cap | seed 1338 | seed 1339 |
|---|---|---|---|---|
| control | false | 0.0 | 4.3356 | 4.3148 |
| **qk** | **true** | **0.0** | **4.2435** | **4.2382** |
| cap | false | 30.0 | 4.3485 | 4.3217 |
| both | true | 30.0 | 4.2517 | 4.2405 |

The absolute levels are not comparable to the 2026-09-22 ablation (different
data snapshot and seeds); only the paired deltas are.

| contrast | seed 1338 | seed 1339 | mean | paired spread |
|---|---|---|---|---|
| **qk − control** | **−0.0921** | **−0.0766** | **−0.0844** | 0.0155 |
| cap − control | +0.0129 | +0.0069 | **+0.0099** | 0.0060 |
| both − control | −0.0839 | −0.0743 | **−0.0791** | 0.0096 |

Control spread at this step pair: **0.0208**; the QK effect is ~4× it, and both
seed deltas are negative.

## The registered bars

| ID | bar | measured | verdict |
|---|---|---|---|
| **Q-1** | bundle mean ≤ −0.020 | **−0.0791** | PASS |
| **Q-1b** | both seed deltas < 0 | −0.0839 / −0.0743 | PASS |
| **Q-2a** | qk mean ≤ −0.010 | **−0.0844** | PASS |
| **Q-2b** | cap mean ≤ −0.010 | **+0.0099** | **FAIL** |
| **Q-3** | both ≤ better component + 0.010 | −0.0791 ≤ −0.0744 | PASS |
| **Q-4** | paired spread reported | 0.0155 / 0.0060 / 0.0096 | — |

## What this establishes

**The −0.08 is real, and it belongs to QK-Norm.** QK-Norm alone reproduces the
original bundled increment almost exactly: **−0.0844** against D8's −0.0834, at
both fresh seeds, with a spread of 0.0155. The bundle's −0.0791 is the same
effect, slightly diluted.

**The logit soft-cap earns nothing.** Alone it is **+0.0099** (both seeds
positive, i.e. slightly worse than control), and adding it to QK-Norm changes
the result by +0.0053 — inside the 0.01 interaction margin, but in the
direction of cost, not gain. Q-2b fails by a wide margin. At τ=30 the logits
rarely reach the cap at this scale, which is the recorded hypothesis for why.

**The recipe can be simpler.** Future configs should drop `logit_softcap` and
keep `qk_norm`. The shipped v0.2.0 is not retro-changed: it was trained with
both, and the weights are what they are. The card claim changes from "QK-Norm +
soft-cap: a further −0.08" to "QK-Norm: −0.084 at two seeds; soft-cap not
supported (−0.010 bar, measured +0.010)".

This is the first shipped recipe ingredient this project has re-tested at two
seeds since D48; it is also the first one whose attribution changed without the
effect disappearing.

## Artifacts

- `val_log-<condition>-s<seed>.csv`, `train_log-…` — the eight curves.
- `configs/` — the eight configs the kernel built; they differ from
  `micro-muon` only in `model.qk_norm`, `model.logit_softcap` and `train.seed`.
- `run_manifest.json` — code fingerprint, input hashes, seeds, bars.
- `x31_verdict.json` — the kernel's computed verdict.
- `kalia-x31-qk-isolation.log` — the console log.

Reproduce the verdict with
`python tools/x31_verdict.py --dir docs/results/x31`.
