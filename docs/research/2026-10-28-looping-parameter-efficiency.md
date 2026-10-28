# Looping — the cheap test was neutral; the parameter-efficiency test is missing

Status: active research record
Date: 2026-10-28
Sources: `docs/research/2026-09-23-adoptable-frontier.md`;
`docs/journal/2026-09-23.md`; Ouro/LoopLM (arXiv 2510.25741, 2025);
Geiping et al., recurrent depth (arXiv 2502.05171, 2025); "Loop, Think, &
Generalize" (arXiv 2604.07822, 2026); D24, D48, D62, D63.

## Correction to the record

On 2026-09-23 the programme promoted **looped / recurrent-depth transformers**
to the front of the architecture queue as **X11** (D24). The design was:
5 unique layers × 2 loops vs 10 unique layers at equal effective depth, then
10 × 2 vs 10 × 1. The evidence was Ouro/LoopLM: 1.4B/2.6B models match up to
12B standard models, with gains in knowledge manipulation (composition,
multi-hop) rather than storage; accuracy scales as log(effective depth), and
depth extrapolation at inference works.

It would be wrong to say X11 was never run. The **simple** version was run in
the same day's architecture ablation:

| arm | params | val @ 700 | Δ vs control |
|---|---:|---:|---:|
| control (Muon + QK-Norm) | 29,921,280 | **3.4924** | — |
| loop2 (6 layers × 2 loops) | 29,921,280 | 3.5012 | **+0.009** |
| deepthin | 28,787,840 | 3.6969 | +0.205 |
| GQA | 28,741,632 | 3.5031 | +0.011 |

The pre-registered promotion rule was ≤ control − 0.02. Looping was therefore
called **quality-neutral** and did not enter v0.2.0. That was the correct
reading of that experiment: at equal parameters, a second pass over the same
six blocks bought nothing at 30M / 700 steps.

## What that experiment did not test

The 6 × 2 vs 6 × 1 comparison holds parameters fixed and doubles compute. It
can only show whether extra compute at the same parameter count helps. It
cannot show whether looping is **parameter-efficient**, because there is no
larger dense model in the comparison.

D24's stronger claim is the one Ouro/LoopLM makes: effective depth can be
decoupled from parameter count. The missing comparison is:

| arm | unique layers | loops | effective depth | params | compute |
|---|---:|---:|---:|---:|---:|
| control | 6 | 1 | 6 | 30M | 1× |
| loop2 | 6 | 2 | 12 | 30M | 2× |
| **deep12** | 12 | 1 | 12 | **~60M** | 2× |

If **loop2 (30M) matches deep12 (60M)** on val, looping recovers a 2×
parameter model at half the parameters. That is a real parameter-efficiency
result even though loop2 is slightly worse than the 30M control. It is exactly
the Ouro claim at our scale, and it has never been run.

## Why it is still the right next architecture bet

1. **It is a genuine architectural change, not a head.** The same parameters
   are applied twice; effective depth is decoupled from parameter count. The
   6-layer micro is shallow, and depth is the cheapest missing ingredient to
   test.
2. **It has a scaling story.** Ouro/LoopLM reports 1.4B matching 12B standard
   models on composition/manipulation; Geiping et al. scale recurrent depth to
   3.5B / 800B tokens. This is not a 30M-only trick.
3. **The code exists.** `GPTConfig.n_loops` is implemented as a repeated pass
   over the same blocks; `configs/micro-loop2.yaml` is a working starting
   point. No new block, optimizer or data pipeline.
4. **The protocol now exists.** D48's objection was variance. X36/X37 built the
   paired-seed, parameter-matched protocol that can test a 0.05-nat effect.
5. **It cannot fail for SUTRA's reason.** SUTRA added a discrete router and
   failed to specialize. Looping adds computation without a router, so a null
   would mean depth itself is not the lever, not that the router collapsed.

## The test to register (X11 parameter-efficiency)

At 30M, 6 layers × 384 wide, same data and steps as the X37 micro, paired
seeds 1338/1339:

- **L-1 (parameter efficiency):** loop2 val ≤ deep12 val + 0.005, both seeds.
  If true, 30M of looped parameters match ~60M of dense parameters.
- **L-2 (quality guard):** loop2 val ≤ 6 × 1 control val + 0.02, both seeds.
  The earlier +0.009 passes this; a larger regression kills the arm.
- **L-3 (depth extrapolation):** with loop2 trained at 2 loops, evaluate at
  1/2/4 loops; 4 loops should not be worse than 2 by more than 0.02. This is
  the literature's key claim; overthinking is the failure mode.
- **L-4 (stability):** no NaN, gradient norm within the control's range, and
  no divergence after the first 100 steps.
- **L-5 (retention):** the existing entity/name retention probe should not
  regress by more than 0.005 versus the control. If it improves, that is the
  composition/manipulation signature the Ouro results predict.

Cost: ~6-8 GPU-h for two seeds × three arms at 30M / 763 steps, because loop2
and deep12 each cost roughly twice the control. If L-1 passes, scale to 58M:
10 × 2 (58M) vs 20 × 1 (~116M), ~15-20 GPU-h per seed pair.

## Risks

- **Compute confound.** loop2 and deep12 both use ~2× the control's compute.
  L-1 compares equal effective depth and equal compute, but unequal parameters;
  that is the point. Do not report L-1 as a compute-efficiency win.
- **30M is below the published scale.** Ouro/LoopLM is 1.4B; the 30M test is a
  screen. A null does not close looping at 1B, but it does close the cheap
  version.
- **Short training.** 763 steps may be too short for the second pass to learn.
  If the 763-step screen is flat, the fallback is 1,500 steps, not an
  immediate scale-up.
- **Stale config.** `micro-loop2.yaml` predates D55 (logit soft-cap removed)
  and uses seed 1337. The new config must use the X37 micro's settings
  (`qk_norm: true`, `logit_softcap: 0.0`, seeds 1338/1339) so the control is
  comparable.
- **D48's lesson.** Two seeds and a paired control are mandatory; a one-seed
  win is not a result.

## Decision

Register X11 parameter-efficiency for the 2026-10-31 quota window. Do not run
another 30M rule-grammar or memory micro before it. If L-1/L-2 fail, the next
candidates are memory layers at 134M (the smallest scale in Berges et al.) and
hyper-connections at 30M; if they pass, scale to 58M and then 1B.
