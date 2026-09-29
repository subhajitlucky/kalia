# v0.2.0 release gate — PASSED

Run 2026-09-29 on `kalia-v020-export` (Kaggle CPU). Full log: the kernel output.

The training checkpoint is **not** a loadable model repo: it carries optimizer
state, internal tensor names, and no config. So the release path is a conversion
followed by a check, and the check is the point.

## The gate

`tools/export_hf_model.py` converts the step-4,770 checkpoint and then **re-scores
the written `model.safetensors`**, not the checkpoint it came from, under the
protocol registered in
`docs/preregistrations/2026-09-28-v020-final-evaluation.md` before any result
existed.

```json
{
  "val_loss": 2.8248,
  "bpb": 0.9244,
  "tokens": 819200,
  "registered": 2.8248,
  "matches": true,
  "step": 4770,
  "train_tokens": 2500853760
}
```

| check | result |
|---|---|
| checkpoint step | 4770, as expected |
| tokens seen | 2,500,853,760 |
| tensors written | **93** — the same layout `kalia-v012` already ships |
| tensors bitwise-identical to the checkpoint | **yes, all 93** |
| safetensors metadata | `step: 4770`, `tokens: 2500853760`, `format: pt` |
| loads into the repo's own `model.py` | clean, no missing or unexpected tensors |
| `lm_head` tied to `tok_emb` in the published file | yes |
| **re-scored under the registered protocol** | **2.8248 / 0.9244 over 819,200 tokens** |
| **matches the pre-registered value** | **yes, to four decimals** |

**PASS.** The artifact that would be published scores exactly what was
pre-registered. That is the only claim that makes a release trustworthy, and it is
the claim a naive conversion silently breaks.

## Two things this exercise found

**safetensors cannot serialise a tied model.** KALIA ties `lm_head` to `tok_emb`,
so those entries are the same tensor. `save_file` raises `RuntimeError: A
potential way to correctly save your model is to use save_model`. The obvious wrong
fix is to drop the head — which would leave the published repo with a *different
tensor set* from the one that was measured. The exporter keeps the head and clones
the alias, preserving values bitwise. `test_export_hf_model.py` asserts the
shared-storage premise first, so the de-share cannot quietly become dead code.

**The published config is derived, not transcribed.** It comes from the
checkpoint's own `config['model']`, with a test asserting that `nope_interval`,
`branch_norm` and `gated_residual` cannot leak into a config that never used them.
An architecture that drifts from the trained one is the failure this whole
exercise exists to prevent.

## Still outstanding

The five small files and `model.safetensors` are produced and verified in the
kernel, but **nothing has been uploaded to `kalia-lm/kalia-v020` yet and the repo
is still private.** The next step is for the same kernel to push the artifact it
just verified, so the file that is published is provably the file that scored
2.8248 rather than a copy that travelled through a laptop.

`docs/public/hf-model-card-v020.md` is written and states plainly that this
release **fails its own pre-registered stability rule on two of five benchmarks**
(ARC-Easy −3.80, LAMBADA −2.20) while improving held-out loss by 0.2285 nats, and
that `kalia-v012` remains the recommended checkpoint.
