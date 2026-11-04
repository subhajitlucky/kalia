# V032 — frozen-base output-adapter fine-tune

Status: **val bar passed (V-1/V-2/V-3/V-4); 2k regression guard pending.**
This is engineering, not a new mechanism. The base v0.3.1 is frozen; only a
zero-init output adapter trains.

## Result

| seed | val | base val | delta | pass |
|---|---:|---:|---:|---|
| 1401 | **2.7493** | 2.7580 | **−0.0087** | yes |
| 1402 | **2.7495** | 2.7580 | **−0.0085** | yes |

- Base val re-measured in the same kernel: **2.7580** (published 2.7579).
- Bar: adapter ≤ base − 0.005 both seeds. Both pass.
- Frozen-weight checksum: **unchanged** before/after, both seeds.
- Adapter: `512 → 1024 → 512`, **1,048,576** trainable params; no base updates.
- 2,000 steps, micro 8 × grad accum 8, LR 1e-3, cosine; 2 seeds.
- Base sha `bbd8a036…`; kernel `subhajitlucky/kalia-v032-adapter` v2,
  revision `c40b464`.

## Verdict

**V-1, V-2, V-3, V-4 pass; `advance = true` for the val bar.** The adapter is
an internal **v0.3.2 candidate on GitHub only**. No Hugging Face push.

The 2,000-sample benchmark pair (V-5) is the conditional regression guard. If
any task regresses beyond 2σ versus v0.3.1, the candidate is withheld. If it
does not regress, the candidate is recorded as a small val improvement, not a
new mechanism.

## Artifacts

- `v032_summary.json` — full result, hashes, seeds, frozen check.
- `v032_train_log.csv`, `v032_val_log.csv` — training/val curves.
- `adapter_seed1401.pt`, `adapter_seed1402.pt` — adapter state dicts only
  (~4.2 MB each); the base is the public v0.3.1.
- `v032.log` — kernel log.
- Pre-registration: `docs/preregistrations/2026-11-04-V032-output-adapter.md`
  (hash `4ead53b6…`).

## What this is not

- Not a new architecture, attention, tokenizer or pretraining objective.
- Not a release candidate until V-5 passes.
- Not a Hugging Face update.
