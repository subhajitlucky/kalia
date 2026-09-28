# Pre-registration — X18: Two 2026 Frontier Changes (NoPE, Gated Residual)

Registered: 2026-09-28, before any run of this experiment exists.
Related: D5 (screen on benchmarks, not loss alone), X17 (doc-mask),
`2026-09-28-why-27b-beats-397b.md`, `2026-09-28-open-source-landscape-and-transfers.md`.

## Why these two

Both come from the frontier survey, and both are cheap enough to screen honestly.
Neither is a large architectural gamble: they are the two changes with published
evidence that survive a cost check at 58M.

- **NoPE** — SmolLM3 (3B): drop rotary embeddings on every 4th layer. Their
  ablations on 3B/100B tokens found long-context gains at no short-context cost.
  The property we like: it *removes* machinery and is exactly parameter-neutral
  (verified: 29,920,512 with and without at micro dims), so a difference in
  result cannot be explained by capacity.
- **Gated Residual** — Qwen3.8-Flash-Next: normalise residual branches
  independently and read them through a data-dependent elementwise gate. Their
  reported split: widening the stream +1.58 accuracy, data-dependent read a
  further +1.98, while the *loss* gap was 0.002.

**Scope, stated before the run:** we implement only the **gating half** of
Gated Residual, on the existing width. The widening (four full-width branches) is
the expensive part and is not tested here. This arm therefore tests the
*mechanism*, not the capacity increase, and is named `gated` accordingly.

## Arms

| Arm | Config | Change | Params |
|---|---|---|---|
| **control** | `micro-base` | none | 29,920,512 |
| **nope** | `micro-nope` (`nope_interval: 4`) | RoPE dropped on layers 4 and 8 | 29,920,512 (identical) |
| **gated** | `micro-gated` (`gated_residual: true`) | gated residual read | 30,072,768 (+0.5%) |

Same 30M architecture otherwise, same seed 1337, same data, same 500 steps, same
LR schedule. One variable per arm, three arms total.

## Pre-registered predictions

| ID | Prediction | Threshold |
|---|---|---|
| **F-1** | NoPE matches or beats control on val loss | val loss ≤ control |
| **F-2** | NoPE costs no forward quality while helping long-range | forward val within +0.01, LAMBADA ≥ control |
| **F-3** | Gated residual improves accuracy by more than it improves loss | val loss ≤ control **and** ≥0.5 pp better on a downstream task |
| **F-4** | The gate actually opens during training | mean gate value at the end > 0.05 |

**F-3 is the important one.** It is the only prediction here that tests a claim
about *our own methodology* rather than a technique: Qwen reported a 0.002 loss
gap alongside a multi-point accuracy gap. If we reproduce that pattern at 30M, it
is direct evidence that D5 (select on benchmarks, not loss) was right — and it
would be the first result in this project that is about how to do science on a
58M model rather than about the model.

## Honest priors, recorded before the run

- **NoPE is the safer bet.** It is parameter-neutral and validated at 3B, but
  3B → 30M is a 100× scale drop and the ablation window here is 500 steps.
- **Gated residual is the more interesting bet and the likelier to be noise.** At
  30M with a rank-32 gate added to a 384-dim stream, the mechanism may simply be
  too small to matter. A null here is uninformative about Qwen's result at 125B.
- **Both arms will lose to control on absolute loss** if anything: the gate starts
  near-closed (mean 0.018) and must learn to open it, which costs early steps.
  We therefore judge on the *final* loss, not the curve shape.

## Decision rule (fixed now)

- **F-1 passes** → NoPE is a candidate for v0.3.0; it is free and safe.
- **F-3 passes** → the loss/accuracy divergence is reproduced at our scale, which
  is itself a publishable methodological result, independent of whether the
  mechanism ships.
- **Both fail** → published as a negative result, with the caveat above that a
  null at 30M says little about 125B.

## Cost

~1.5h GPU for three arms at 30M / 500 steps. Scheduled **before** session 3 of
v0.2.0 so it uses the free batch-GPU slot that session 2's completion released,
rather than delaying the main run. Total remaining quota after both: ~2.3h spare.

## Deviations

Any change to arms, steps, seed, thresholds, or metrics after this commit
requires a hash-registered amendment. Results are reported regardless of outcome.
