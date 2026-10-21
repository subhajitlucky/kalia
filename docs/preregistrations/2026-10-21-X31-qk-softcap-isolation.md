# Pre-registration — X31: does QK-Norm + logit soft-cap earn its place?

Registered: 2026-10-21, before any GPU time is spent. Related: D8, D48, X21,
the 2026-10-18 whole-record audit.

## Why this exists

D8 adopted QK-Norm + logit soft-cap on one micro-ablation, one seed:

| arm | val loss (30M, 763 steps) |
|---|---|
| AdamW control | 3.8041 |
| Muon | 3.5937 |
| **Muon + QK-Norm + soft-cap** | **3.5103** |

The increment credited to the pair is **−0.0834 nats**, measured at seed 1337.
It has never been replicated, and the two components have never been isolated.
The 2026-10-18 audit put the number against the rulers this project has since
measured:

- cross-configuration 30M control spread: **0.1139** — the effect is below it;
- paired control spread at seeds 1338/1339 (X21): **0.0139** — the effect is
  above it.

So the honest state is not "validated" (D8's status) and not "refuted". It is
**one seed, one bundle, measured against the wrong ruler for the claim it is
carrying**. The shipped cards and SYNTHESIS still cite the −0.08, and the recipe
ships both components. This experiment decides whether that citation survives.

This is a **micro-scale recipe test**. Passing does not transfer a claim to the
58M model; it establishes whether the 30M ablation that the cards cite
reproduces, and which component carries it.

## Design

30M parameters (6 layers × 384d, 6 heads, 512 context), Muon recipe, 763 steps,
the v2b `train.bin`/`val.bin` pair already used by X17/X21. One kernel, eight
arms, one machine, one session.

| condition | `qk_norm` | `logit_softcap` | seeds |
|---|---|---|---|
| `control` | false | 0.0 | 1338, 1339 |
| `qk` | true | 0.0 | 1338, 1339 |
| `cap` | false | 30.0 | 1338, 1339 |
| `both` | true | 30.0 | 1338, 1339 |

Seed 1337 is **excluded**: it generated the hypothesis and including it would be
circular. Seeds 1338/1339 are the fresh pair D48 established, and X21 already
measured the control's paired spread on them (0.0139).

All four conditions are built from `configs/micro-muon.yaml`; the only keys that
move are `model.qk_norm`, `model.logit_softcap` and `train.seed`. The control
sets both flags to the model's defaults explicitly so all four configs have the
same key set. A test asserts this, and the kernel re-asserts it before training.

**Sequential, not interleaved.** Conditions run control → qk → cap → both within
each seed, then the next seed. They share a machine and a session, but they do
not alternate across many runs the way the modded-nanogpt records do. The
write-up must say that, not imply a stronger design.

## Pre-registered predictions

| ID | Prediction | Threshold |
|---|---|---|
| **Q-1** | The bundled claim replicates | mean paired Δ(`both` − `control`) ≤ **−0.020** |
| **Q-1b** | It is not one lucky seed | **both** individual seed deltas < 0 |
| **Q-2a** | QK-Norm alone earns its place | mean paired Δ(`qk` − `control`) ≤ **−0.010** |
| **Q-2b** | Soft-cap alone earns its place | mean paired Δ(`cap` − `control`) ≤ **−0.010** |
| **Q-3** | The bundle is not antagonistic | mean Δ(`both`) ≤ min(mean Δ`qk`, mean Δ`cap`) + **0.010** |
| **Q-4** | Paired-difference spread | reported, no threshold |

Q-1's −0.020 is just above the measured paired control spread (0.0139) and a
quarter of the originally reported 0.0834. It is deliberately not −0.083: a
replication is allowed to be smaller. Q-2a/Q-2b use −0.010 because a single
component is expected to carry less than the bundle; that is the smallest
increment this project has treated as a mechanism (D15's 0.02 was for
promotion, not detection).

## Honest priors, recorded before the run

- **QK-Norm is the more likely of the two to survive.** It is standard practice
  and the mechanism is well understood. Soft-cap at τ=30 may be inert at this
  scale, where logits rarely reach the cap; if it is, the recipe gets simpler.
- **The bundle may replicate while neither component clears Q-2.** Two small
  effects can add up; that is an interaction, not evidence for either
  component's mechanism. Report it as such.
- **The most likely single outcome is a smaller effect than 0.0834.** A
  single-seed micro-win that large is usually partly seed luck.
- **Two seeds is the minimum honest test.** If Q-1 and Q-1b pass, the result is
  "replicated at two seeds", never "confirmed".

## Decision rule (fixed now)

- **Q-1 and Q-1b pass** → the bundled claim is replicated at two seeds. Keep
  both components in the recipe. Report Q-2a/Q-2b as the attribution, and Q-3
  as the interaction check.
- **Q-1 passes, Q-1b fails** → one seed carries it. Report as unreplicated; do
  not strengthen the card claim.
- **Q-1 fails** → the −0.08 is not replicated. Withdraw it from the cards,
  SYNTHESIS and D8's "validated" status; record D8 as **not replicated at two
  seeds**. No further QK/soft-cap work at this scale.
- **Q-1 passes, one component fails its Q-2 bar** → keep the bundle for the
  shipped model, but record the inert component as unexplained and remove it
  from the *next* recipe unless a mechanism is found. Do not retro-change
  v0.2.0.
- **Q-3 fails** → the combination is antagonistic despite the bundle's mean.
  Report both; do not ship the bundle.

## Cost

Eight arms × 763 steps at 30M. X17 measured four 500-step arms at **1.22h**
including setup and evaluation, so eight 763-step arms are **≈3.7 GPU-hours**,
inside the remaining quota with a margin. The entity-holdout measurement runs
on CPU at the same time and does not compete for GPU quota.

## Deviations

Any change to arms, seeds, steps, thresholds, or metrics after this commit
requires a hash-registered amendment. Results are reported regardless of outcome.

---

## Result (appended 2026-10-21, after the run)

Kernel `subhajitlucky/kalia-x31-qk-isolation`, COMPLETE, 3.8 GPU-h. All eight
arms reached step 763; the eight val logs end at step 700, the last scheduled
eval before the stop.

| condition | seed 1338 | seed 1339 | mean Δ vs control |
|---|---|---|---|
| control | 4.3356 | 4.3148 | — |
| **qk** | **4.2435** | **4.2382** | **−0.0844** |
| cap | 4.3485 | 4.3217 | **+0.0099** |
| both | 4.2517 | 4.2405 | −0.0791 |

- **Q-1 PASS** — bundle mean −0.0791, bar −0.020.
- **Q-1b PASS** — both seed deltas negative (−0.0839 / −0.0743).
- **Q-2a PASS** — QK-Norm mean −0.0844, bar −0.010.
- **Q-2b FAIL** — soft-cap mean +0.0099, and both seeds positive.
- **Q-3 PASS** — bundle −0.0791 is within 0.005 of QK alone, inside the 0.010
  interaction margin.
- **Q-4** — paired spreads: qk 0.0155, cap 0.0060, both 0.0096. Control spread
  at this step pair is 0.0208; the QK effect is ~4× it.

**Decision.** The −0.08 is real and belongs to **QK-Norm**; the logit soft-cap
adds nothing and is slightly worse than control. Future configs should keep
`qk_norm` and drop `logit_softcap`. The shipped v0.2.0 is not retro-changed.
The public claim changes from "QK-Norm + soft-cap: a further −0.08" to
"QK-Norm: −0.084 at two seeds; soft-cap not supported (+0.010 measured)".

Artifacts: `docs/results/x31/`. The computed verdict is in
`docs/results/x31/x31_verdict.json` and is reproducible with
`python tools/x31_verdict.py --dir docs/results/x31`.
