# X17 — intra-document masking: raw artefacts and the registered verdict

Registered `docs/preregistrations/2026-09-28-X17-doc-mask.md`; design amended
`2026-10-18-X17-amendment-1.md` (paired two-seed, DM-1b concordance, DM-5 paired
spread). Kernel `subhajitlucky/kalia-x17-docmask`, code `5552e54dd68c`. One
kernel, four arms, sequential with control and treatment alternating within each
seed, 30M parameters, 500 steps, ~69 minutes of GPU.

| arm | seed 1338 | seed 1339 |
|---|---|---|
| control (`micro-base`, `doc_mask: false`) | **4.8773** | **4.8920** |
| treatment (`micro-docmask`, `doc_mask: true`) | 4.9253 | 4.9328 |
| paired delta | **+0.0480** | **+0.0408** |
| eval points ahead | 0 / 5 | 0 / 5 |
| step time | 1.971 s → 2.079 s (×1.054) | 1.976 s → 2.078 s (×1.052) |

## The registered bars

| ID | bar | measured | verdict |
|---|---|---|---|
| **DM-1** | mean paired Δ ≤ −0.010 | **+0.0444** | **FAIL** |
| **DM-1b** | both deltas negative | both **positive** | **FAIL** |
| **DM-2** | ahead at ≥3 of 5 points, both seeds | 0 of 5, both seeds | **FAIL** |
| **DM-3** | treatment step time ≤ 1.25× control | ×1.054 | PASS |
| **DM-4** | mask engagement < 1.0 | **0.7266** | PASS |
| **DM-5** | paired-difference spread (reported) | 0.0072 | — |

**REGISTERED VERDICT: do not promote.**

The effect is not small relative to the ruler: 0.0444 against a paired spread of
0.0072 is about **6×** it, consistent in sign at both seeds. Masking does
something. The question is what.

## The confound, and why the numbers above are not the finding

Both arms are evaluated with **full causal attention and no document mask** —
`evaluate()` calls `dataset.get_batch(8, device)` without masks and `model(x, y)`
without an `attn_mask`. So:

- control trained **unmasked** → evaluated unmasked — **matched**
- treatment trained **masked** → evaluated unmasked — **mismatched**

The treatment is being tested in a regime it was not trained for. The
registration saw this coming and said so, in its own decision rule:

> *"DM-1 and DM-2 pass → promote to v0.2.1, and re-baseline the canonical val
> number, because **the eval must use the same masking as training to be
> meaningful**."*

It appears only in the pass branch, as a reason to re-baseline afterwards. The
experiment as designed never applies it, and neither did the amendment.

**So the defensible statement is narrower than "masking hurts":** *training with
document masking and evaluating without it costs 0.0444 nats at 30M / 500 steps,
at 5% throughput.* Whether masking helps or hurts needs an evaluation that applies
the same mask the model trained under — the 2×2 of train-time × eval-time masking
on the four archived checkpoints, which is CPU-only and quota-free.

That is the next step, and it should happen before this result is written up as a
negative. Recording a confounded number as a mechanism finding is the defect this
project has spent the day removing from its own record.

## The 2x2, and the reversal

`kalia-x17eval-mask2x2` (CPU) scored all four archived checkpoints under both
evaluation settings — `tools/kalia_x17_eval_kernel.py`, manifest `mask2x2.json`.
Identical windows (a fresh `torch.Generator` seeded 1234 per cell, where
train.py's `evaluate()` draws from the global RNG) and identical loss tokens
(`loss_mask` never passed, since a document mask also drops boundary positions
from the loss and would make the columns incomparable). Only `attn_mask` varies.

|  | eval unmasked | eval masked |
|---|---|---|
| **train unmasked** (control) | 4.8037 | 4.8122 |
| **train masked** (treatment) | 4.8484 | **4.7998** |

Paired deltas, treatment − control:

| seed | eval unmasked | eval masked |
|---|---|---|
| 1338 | +0.0493 | −0.0088 |
| 1339 | +0.0400 | −0.0160 |
| **mean** | **+0.0446** | **−0.0124** |

- The mismatched mean **+0.0446 reproduces X17's recorded +0.0444** under fixed
  windows. The measurement was right; the reading of it was not.
- Matched — the cell the registration was missing — the treatment is **better**
  by 0.0124, negative at both seeds.
- **Interaction −0.0570.** The train/eval mismatch was worth more than the effect
  it was hiding.
- The control's own shift under eval-masking is +0.0085: masking a model that
  never trained with it costs it. That is the mechanism of the artifact.

**X17's registered verdict stands as measured: DM-1 FAIL on the protocol as
registered.** Nothing here rewrites it. What this establishes is that the number
it failed on was the mismatch, and that under matched evaluation the contrast
changes sign and would clear the −0.010 bar at −0.0124.

Caveats, before this is used for anything:

- **Different protocol.** 100 batches with a fixed seed, against the registered
  20 drawn from the global RNG. A re-measurement, not the registered verdict.
- **DM-2 is not testable here.** "Ahead at ≥3 of 5 eval points" needs the curve;
  this scores final losses only.
- **At the bar, and the seeds differ 2×** (−0.0088 against −0.0160). A 0.0124 mean
  against a 0.010 bar is not comfortably above it.
- **The manifest's `code_fingerprint` is not comparable with X17's**
  (`c3eb9571…` against `c1b861d3…`): this kernel's first version grew its own
  cruder fingerprint — a hash of top-level `*.py` bytes rather than the runners'
  per-file, path-qualified digest — so it fingerprints the same bundle
  differently. Fixed in the kernel; the archived value stands as wrong, and is
  recorded rather than quietly regenerated.

## Artifacts

- `val_log-<arm>-s<seed>.csv`, `train_log-<arm>-s<seed>.csv` — per arm
- `run_manifest.json` — code fingerprint `c1b861d3…` (identical to the day's
  no-replay runs, so code-identity is proven), `train.bin` `313fc7ba…`,
  `val.bin` `625f8e5f…` (matches `docs/legal/val-shard-hashes.json`), and DM-4's
  engagement measured **before** training
- `mask2x2.json` — the eight evaluation cells, checkpoint hashes, and the windows
  and loss-token policy the comparison rests on
- `kalia-x17-docmask.log`, `kalia-x17eval-mask2x2.log` — the kernel logs

Reproduce with `python tools/x17_verdict.py` and `python tools/x17_mask2x2.py`.
