# Entity-verified decoding — the holdout test

The project's one measured inference-time win was entity-verified decoding:
retention **0.0907** against greedy's **0.0296**, +0.061, with the human
reference at **0.0978** (2026-10-11). It was the basis for the memory-only rung.

It is also close to circular. `entity_decode.select_best_index` picks the
candidate maximising `(retention, mean_span, logprob)`, where `retention` comes
from `eval_entities` — and `eval_entities.retention` is the metric the kernel
then reports. The measurement is "a selector that maximises X achieves a high X".

This run breaks the circularity. Every generated continuation is split at its
midpoint; the selector sees only the **first half**, and retention is measured on
the **second half it never saw**:

    select on   retention(prompt, first_half)
    report      retention(prompt, second_half)

If selecting on the first half lifts retention in the second half above greedy,
entities persist further than where the rule looked. If it does not, the 3× was
the selector marking its own paper.

## Protocol

20 frozen Gutenberg stories (`eval/ttt_probe_stories.json`), prompt 768 tokens,
256 generated, 8 samples at temperature 0.8 / top-k 200, plus one greedy draw.
The same samples are ranked four ways, paired per story, with a 5,000-resample
bootstrap over stories. The test was fixed in `kalia_entity_holdout_kernel.py`
before the run.

**Device deviation.** The measurement was designed for CPU so it would cost no
GPU quota. It does not fit there: `generate()` has no KV cache and re-runs the
full 768-token context for every one of 256 tokens. The first successful run
reached 11/20 stories before cancellation; the 2026-10-21 CPU rerun spent
~70 minutes on story 0 alone. Twenty stories on CPU would exceed the 12-hour
session limit, so the completed run uses the same prompts, N, tokens and seeds
on GPU, with gradients disabled. The device changes the sampled continuations,
so `full_pick` is a re-measurement, not a reproduction of the CPU number; the
paired holdout comparison is unaffected. Manifest: `run_manifest.json`
(`device: cuda`, 20 stories, split 128).

## Results

| selector | holdout retention (unseen half) | full-continuation retention |
|---|---|---|
| greedy | 0.0170 | 0.0296 |
| log-prob pick | 0.0103 | 0.0353 |
| **half pick (the test)** | **0.0277** | 0.0864 |
| full pick (the recorded rule) | 0.0398 | **0.0907** |
| oracle (best holdout among the 8) | 0.0618 | — |
| human reference | 0.0803 | 0.0978 |

Paired bootstrap over the 20 stories:

| contrast | mean | 95% CI | verdict |
|---|---|---|---|
| **half pick − greedy, holdout** | **+0.0108** | **[−0.0073, +0.0318]** | **INCLUDES ZERO** |
| half pick − log-prob, holdout | +0.0175 | [+0.0027, +0.0361] | excludes zero |
| full pick − greedy, full text | +0.0612 | [+0.0389, +0.0860] | excludes zero |

## What this establishes

**The recorded 3× is a self-selection effect.** When the selector and the metric
see the same continuation, the rule scores 0.0907 against greedy's 0.0296 — the
published number, reproduced. When the selector is shown only the first half and
scored on the second, the advantage falls to **+0.0108** with a confidence
interval that includes zero. The rule does not reliably find entity-preserving
continuations beyond the point where it looked.

**What survives is coverage.** The oracle — the best of the same 8 samples,
scored on the held-out half — reaches **0.0618** against greedy's 0.0170. The
good continuation is in the sample set; the model's own probabilities cannot
find it (log-prob pick scores 0.0103, below greedy on the holdout half). That is
a statement about selection being the bottleneck, not about the entity rule
being the solution.

**The half-pick does beat log-prob-pick by +0.0175**, with a CI excluding zero.
That is real but small, and it is not a win over greedy: log-prob is *below*
greedy on the held-out half. The comparison that matters for the memory-only
rung is half-pick − greedy, and it is null at this n.

## Decision

**Do not cite the 3× as evidence for the memory-only rung.** The rung's measured
basis is now the coverage result — the right continuation is in the pool — not
the entity rule. The open question is a selector that generalises to text it did
not use to select, and the current rule is not it.

The 2026-10-11 result remains in the record as measured, with this correction
attached. The 2026-10-17 programme direction's line "entity-consistent decoding
(+0.061 retention, 3× greedy, 0 GPU-h)" must be read with the holdout result
beside it; the 0-GPU-hour claim is also no longer true of the completed
measurement (the CPU protocol could not finish).

## Artifacts

- `entity_holdout.json` — aggregates, per-story rows, bootstrap tests.
- `run_manifest.json` — device, code fingerprint, checkpoint hash, story hash.
- `kalia-entity-holdout.log` — the run's console log.

Kernel `subhajitlucky/kalia-entity-holdout` v5, code `3b522ee`. The earlier
CPU run was canceled at 11/20; its partial is not used here.
