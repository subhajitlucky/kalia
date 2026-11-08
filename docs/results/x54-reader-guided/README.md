# X54 — reader-guided diffusion consistency: superseded

Status: **superseded by X55.** The baseline diagnostic showed the
see→draw→see loop already closes with the embedding the v1 diffusion was
trained with, so the registered reader fine-tune was not needed. The run
stopped on an evaluation label-slice bug before the fine-tune started.

## What happened

The X54 kernel loaded the v1 diffusion and the frozen v2.1 captioner and ran
the baseline cycle using the **bidirectional masked-mean** embedding from v1
training. On 500 held-out and 500 seen real images:

| baseline (raw / corrected) | held-out | seen |
|---|---:|---:|
| loop exact | 0.966 / **0.943** | 0.966 / **0.943** |
| source readback | 0.972 / **0.949** | 0.972 / **0.949** |
| real caption | 1.022 / **0.998** | 1.020 / **0.996** |

The raw counts exceed 1.0 because the last batch was not truncated to `n`:
512 images were processed and divided by 500. The corrected values are the raw
counts multiplied by 500/512. The counter bug is fixed in the committed kernel.

The classifier step then failed with a shape mismatch (512 generated images vs
500 labels), caused by the same last-batch bug. The run was not repeated: the
baseline already showed that the loop closes with the v1 training embedding.
X55 then ran the clean A/B and established the cause.

## Why no reader fine-tune

X54 was registered to close an open loop by adapting the generator to the
reader. The baseline showed the loop was not open when the diffusion was
conditioned correctly; the earlier cycle null was an inference-embedding
mismatch. Fine-tuning the generator would have answered a question that no
longer existed, so no X54 release and no fine-tune. The X54 pre-registration is
retained; the treatment was not executed because its premise was false.

## What it led to

X55: bidirectional held-out loop **0.930**, causal control **0.000**, with the
same modules. See `docs/results/x55-embedding-fix/README.md` and release
**v0.5.4-vision-cycle**.

## Artifacts

- `x54.log` — baseline output and the label-slice traceback.
- Kernel: `subhajitlucky/kalia-x54-reader-guided-diffusion`.
- Pre-registration: `docs/preregistrations/2026-11-08-X54-reader-guided-diffusion.md`
  (hash `412a8fce…`).
- Corrected kernel: `tools/kalia_x54_reader_kernel.py` (last-batch truncation).
