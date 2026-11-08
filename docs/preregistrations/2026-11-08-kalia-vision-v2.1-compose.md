# Pre-registration — KALIA-Vision v2.1: compositional split

Registered: 2026-11-08, before the split GPU run.

## Question

The v2 captioner saw all 144 two-object combinations. Does it **compose**
unseen shape/color combinations, or memorize the training grid?

## Split

- All combinations: 3 shapes × 4 colors × 3 shapes × 4 colors = 144.
- Hold out a deterministic **20%** of combinations (seed 2026).
- Train on the remaining 80%; validate on seen combinations.
- Test on:
  - **held-out combinations** (unseen pairings of seen attributes),
  - **seen combinations** (within-distribution control).

No external data; no pretrained vision; KALIA frozen.

## Model

Same as v2: from-scratch CNN encoder + 4 image tokens + frozen v0.3.1 LoRA
caption decoder; 8 epochs.

## Bars

| ID | prediction | threshold |
|---|---|---|
| **C-1** | held-out exact caption | ≥ 0.50 |
| **C-2** | held-out attribute accuracy (each) | ≥ 0.80 |
| **C-3** | seen exact caption (sanity) | ≥ 0.90 |
| **C-4** | generalization gap reported | seen − held-out |
| **C-5** | from scratch | no pretrained vision; KALIA frozen |

## Decision rule

- **C-1/C-2 pass:** evidence of compositional generalization in the image
  captioner; release **v0.5.3-vision-compose**.
- **C-1/C-2 fail:** memorization boundary; log it; no tuning after the result.
- Either way, this is the cross-modal binding data point.

## Cost

≤ 1 GPU-h.
