# Pre-registration — X55: text-conditioning embedding fix

Registered: 2026-11-08, before the X55 GPU run.

## Context

The X54 baseline diagnostic produced a surprising result. With the text
embedding used to train the v1 diffusion, the see→draw→see loop on held-out
combinations was **~0.94**, not the **0.032** reported by the earlier cycle
experiment. The X54 counters had a last-batch bug (512 samples divided by 500),
but the raw counts still imply ~0.94.

The difference is the text encoder call:

- v1 diffusion training (`kalia_vision_train_kernel.py`):
  `base_text.encode(x, attn_mask=mask[:, None, None, :])` — a **bidirectional**
  masked mean over the caption tokens.
- earlier cycle kernels: `base_text.encode(ids).mean(1)` — a **causal** mean.

The diffusion's text-conditioning layer was trained on the bidirectional
embedding. The cycle kernels fed it the causal embedding. That is an inference
mismatch, not a compositional failure of the model.

## Question

Does the see→draw→see loop close when the diffusion is conditioned with the
same text embedding it was trained with?

## Design

A/B with everything else fixed:

- same v1 diffusion U-Net (checkpoint `ba2947e8…`);
- same v2.1 captioner (checkpoint `e16cb9cd…`), frozen;
- same real test images, same held-out combination split (29/144, seed 2026);
- same 100-step DDIM sampler.

Only the text-embedding function differs:

- **B** (bidirectional): v1 training embedding, masked mean with
  `attn_mask=mask[:, None, None, :]`.
- **C** (causal): the earlier cycle embedding, `encode(ids).mean(1)`.

B is evaluated on 1,000 held-out and 1,000 seen real images. C is evaluated on
500 held-out and 500 seen as the control. We also report the per-caption cosine
similarity between B and C.

No training, no fine-tuning. All modules are ours: KALIA v0.3.1, the v1
diffusion, the v2.1 captioner, the synthetic data. No pretrained vision, no
external model, no distillation.

## Bars

| ID | prediction | threshold |
|---|---|---|
| **E-1** | B held-out loop exact (c1 == c2) | ≥ 0.70 |
| **E-2** | B held-out loop attributes (each) | ≥ 0.80 |
| **E-3** | B held-out source readback exact | ≥ 0.70 |
| **E-4** | B held-out classifier exact (quality) | ≥ 0.80 |
| **E-5** | B held-out real caption exact (sanity) | ≥ 0.90 |
| **E-6** | C control reported and materially lower than B | diagnostic |
| **E-7** | from scratch; no external weights/distillation; artifacts | pass |

## Decision rule

- **E-1…E-5 pass and C is materially lower:** the earlier see→draw→see null
  was an inference-embedding artifact. Release **v0.5.4-vision-cycle** with the
  corrected embedding function and log the correction.
- **E-1…E-5 fail:** the loop is still open even with the training-time
  embedding; log the boundary. Do not tune after the result.
- **E-6 is not a pass/fail bar**; it is the causal control that distinguishes
  the embedding mismatch from a genuine cross-modal gap.

## Cost

≤ 1 GPU-h.
