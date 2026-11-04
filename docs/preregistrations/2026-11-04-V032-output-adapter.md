# Pre-registration — V032: frozen-base output-adapter fine-tune of v0.3.1

Registered: 2026-11-04, before GPU time. **This is engineering, not a new
mechanism.** The mechanism is a known output adapter; the question is whether a
frozen-base adapter can improve v0.3.1's val without retraining the base.

## Why this exists

The audit found no A-class new mechanism. The only positive val signal in the
programme is X40's parameter-matched output adapter: at 30M it improved val by
0.26 nats over control while the event workspace was 0.44–0.48 nats worse than
the adapter. v0.3.1 is a 57.8M weight merge; a small output adapter may correct
residual output errors without changing the base.

This is **not** a new architecture, pretraining objective, attention or
tokenizer. It is a post-training capacity/placement experiment. The result is a
better checkpoint or a clean null, not a novelty claim.

## Base

`kalia-lm/kalia-v031` (`model.safetensors`, sha `bbd8a036…`, 57,856,256
parameters, val 2.7579, bpB 0.9016). The base is frozen for the whole run.
The safetensors stores both tied embedding copies; loading into `GPT` is fine,
and the frozen-weight hash is checked before and after training.

## Treatment

`OutputAdapter`: `h + down(silu(up(h)))`, `up: 512→1024`, `down: 1024→512`,
`down` zero-initialised. Train **only** the adapter; all base parameters have
`requires_grad=False`.

| setting | value |
|---|---|
| adapter hidden | 1,024 (1,048,576 params) |
| train data | v2b `train.bin`, context 1,024 |
| steps | 2,000 |
| micro batch × grad accum | 8 × 8 |
| optimizer | AdamW, adapter only |
| LR / warmup / schedule | 1e-3 / 100 / cosine to 0.1× |
| weight decay / clip | 0.01 / 1.0 |
| precision | fp16 autocast + GradScaler |
| seeds | 1401, 1402 |
| eval | canonical 100 × 8 × 1,024 tokens, seed 1234 |
| base eval | same code and batches, frozen base |

## Bars

- **V-1 (primary):** adapter val ≤ base val − **0.005**, both seeds. The base
  val is re-measured in the same kernel; the published 2.7579 is the reference.
- **V-2:** both seeds beat the base; report the mean and seed spread.
- **V-3:** no base parameter changes; frozen-weight hash must match before and
  after.
- **V-4:** adapter parameter count exactly 1,048,576; no other trainable params.
- **V-5 (conditional):** if V-1/V-2 pass, run the 2,000-sample benchmark pair
  and require no >2σ regression vs v0.3.1. No HF push in any case.

## Decision rule

- **V-1/V-2 pass:** save the adapter, log a v0.3.2 candidate on GitHub only.
  No Hugging Face upload. A 2k benchmark is the confirmation, not a rescue.
- **V-1 fails:** close the adapter line; log the null. No tuning of LR, steps,
  or adapter size after seeing the result. No HF.
- The base remains v0.3.1 for the public model; the adapter is an internal
  candidate until it clears the bar.

## Cost

~2,000 steps × 8×8×1,024 ≈ 131M tokens per seed; ~1.5–2 GPU-h per seed,
~3–4 GPU-h for two seeds plus eval. Fits the remaining window.

## Artifacts

- `docs/results/v032-adapter/README.md` — result, positive or negative.
- `v032_summary.json`, `v032_train_log.csv`, `v032_val_log.csv`.
- adapter state dict(s) and base/adapter hashes.
- `docs/preregistrations/2026-11-04-V032-output-adapter.md` (this file).
