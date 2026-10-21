# KALIA — the record of the first release

One document that ties the rest together. Everything here links to its primary
source; nothing is asserted here that isn't measured somewhere in this repo.

**What this is:** a 57.9M-parameter language model trained from scratch, on free
Kaggle GPU time, with every decision, failure and negative result written down as
it happened — a dated snapshot of the first seven days, through the v0.1.2 release.
The living record continues in [`journal/`](journal/) and
[`DECISIONS.md`](DECISIONS.md).

---

## 1. What we actually built

**Architecture** — 57,856,256 parameters, decoder-only, pre-norm.

| component | choice | why |
|---|---|---|
| optimizer | Muon on hidden weights, AdamW on embeddings/scalars | micro-ablation: −0.21 nats; Kimi K2 and GLM-4.5 use it |
| attention | QK-Norm + logit soft-cap (τ=30) | X31 2-seed: QK-Norm carries **−0.084**; soft-cap **+0.010** and not additive |
| tokenizer | GPT-2 BPE, 50,257 | off-the-shelf; training one was not worth the stage |
| depth/width | 10 layers × 512 | thin-and-deep was tested and **lost** 0.2 nats |
| precision | bf16, 2× T4 via DDP | 30 GPU-h/week is the entire compute budget |

**Data** — 2.4B training tokens + 10M validation, 60/20/15/5:

| source | share | licence | documents ≥1024 tokens |
|---|---|---|---|
| FineWeb-Edu (dedup) | 60% | ODC-By-1.0 | 26.2% |
| TinyStories | 20% | CDLA-Sharing-1.0 | 0.03% |
| Cosmopedia v2 | 15% | Apache-2.0 | 13.2% |
| Python (per-file filtered) | 5% | 7 permissive licences | — |

**Training** — 4,770 steps, 2,500,853,760 tokens, 6.33h in the final session.
AdamW-style schedule on the embeddings, Muon LR 0.02, `min_lr_ratio 0.1`.

---

## 2. The results

### v0.1.2 → v0.2.0, the same yardstick

The corpus was rebuilt for licence compliance, which redefined the held-out set
(D43), so both models are scored on the canonical v2b set with the identical
100-batch protocol.

**Primary metric — deterministic validation loss:**

| | v0.1.2 | v0.2.0 | Δ |
|---|---|---|---|
| **val loss** | 3.0533 | **2.8248** | **−0.2285** ✅ |
| bits per byte | 0.9992 | **0.9244** | −0.0748 |

That is **4.6× the pre-registered 0.05 nat bar** (P-B, passed).

**Zero-shot benchmarks** (lm-eval, 0-shot, **2,000 samples**; the 2026-10-21
re-measurement. The original 500-sample run remains in the record below.):

| task | v0.1.2 | v0.2.0 | Δ (pp) | z | S-A (≤1.0pp) |
|---|---|---|---|---|---|
| PIQA | 59.63 | 61.37 | +1.74 | 1.08 | ok |
| HellaSwag | 35.60 | 35.60 | 0.00 | 0.00 | ok |
| WinoGrande | 50.28 | 50.75 | +0.47 | 0.24 | ok |
| LAMBADA | 25.15 | 23.90 | **−1.25** | −0.92 | **FAIL** |
| ARC-Easy | 45.95 | 46.40 | **+0.45** | 0.29 | ok |

**Registered verdict at 500 samples: P-B PASS, S-A FAIL on ARC-Easy and
LAMBADA. At 2,000 samples: S-A fails on LAMBADA only.** v0.2.0 therefore remains
an **experiment release, not a promotion** — it still trips the registered 1.0 pp
rule on the higher-resolution measurement, but by 0.25 pp and at z=−0.92.

**Read the noise before the numbers.** At 2,000 samples the standard errors are
~1.0–1.4 pp, and **none of the five movements reaches 2σ** — the largest is PIQA
at z=1.08. The 500-sample ARC-Easy −3.80 did not replicate (**+0.45** at 2,000),
and the 500-sample HellaSwag +3.00 became **0.00**. The honest summary is *a
clear improvement in held-out loss, no individually significant change in
zero-shot accuracy, and one task — LAMBADA — outside the registered allowance by
0.25 pp, not two regressions as the 500-sample run suggested.* The registered
thresholds were not moved to escape this.

> Corrected 2026-10-18: this section previously read "three of the five tasks are
> not separable from v0.1.2 at all" against "1.8–2.2 pp". Only WinoGrande is
> inside one standard error. The registered verdict (S-A FAIL) is unchanged, and
> the same false sentence remains visible in
> `docs/preregistrations/2026-09-28-v020-final-evaluation.md`, which is
> hash-anchored and is not edited.
>
> Re-measured 2026-10-21 at 2,000 samples: the ARC-Easy regression did not
> replicate (+0.45) and the HellaSwag gain did not either (0.00). LAMBADA is the
> only 2,000-sample task outside the 1.0 pp allowance (−1.25, z=−0.92). The
> frozen 500-sample registration is not edited; the higher-resolution result is
> in `docs/eval/2026-10-21-v020-bench-pair-2k.md`.

**The interim measurement was actively misleading.** At step 3,470, four of five
tasks were regressing. The final checkpoint reversed that on three. Conclusions
about the corpus have to come from the final checkpoint.

---

## 3. Strong points

**The loss gain is real and properly registered.** Every threshold was hashed
into `docs/preregistrations/LEDGER.md` before the run. Anyone can recompute the
hash at the registration commit and confirm the prediction existed first —
`python tools/verify_prereg_ledger.py` does exactly that for all 9 entries.

**Negative results are published with the same prominence as wins.** X16's
reversal transform missed both pre-registered bars on both seeds and was
rejected; X18's architecture arm produced a real loss gain we could not
attribute, and saying so cost us the finding rather than keeping it. D42
committed to this in advance.

**The measurement caught our own methods being wrong.** Twice, against
ourselves:
- The corpus rebuild silently redefined the yardstick, and a 0.74-nat "regression"
  was entirely an artifact of comparing across sets (D43).
- X18's accuracy threshold of 0.5 pp sits **4.5× below the benchmark's own
  standard error** of ~2.2 pp. It passed. The pass was worth nothing (D45).

**Data provenance is audited, not assumed.** 41.7% of the code corpus measured
non-permissive (39.5% GPL-family), found in 11 minutes of CPU, and disclosed on the live model card rather
than quietly fixed.

**Reproducibility is real.** A full test suite, a resumable multi-session harness, a
dataset publisher that verifies by round-trip, and a ledger whose hashes anyone
can check.

---

## 4. Weak points — stated plainly

**At 2,000 samples, one benchmark is outside the allowance.** LAMBADA −1.25 pp
against the 1.0 pp rule, at z=−0.92. The 500-sample ARC-Easy −3.80 did **not**
replicate (+0.45 at 2,000), so the earlier "the corpus cost 3.8 points on
ARC-Easy" reading is retracted. The mixture is still the most important open
lever — a 0.2285-nat loss gain produced no 2σ accuracy movement — but it is not
the accuracy catastrophe the 500-sample run suggested.

**The last 31% of training bought nothing.** The val curve is flat from step
3,250 — 2.874–2.940 across the final 1,500 steps. This is the same J2 plateau
that stopped v0.1.2 at 73% of its schedule. We paid 6.3 hours for it.

**Every architecture arm so far has been disappointing.** Looped depth: neutral.
GQA: neutral. Thin-and-deep: worse. Reversal: worse. Gated residual: −0.0436
nats at micro scale, and then the gate turned out to be inert.

**The training loop is wrong for continual learning.** Every run in this
project's history has decayed to `min_lr_ratio 0.1` and stopped. The canonical
continual-pretraining recipe — re-warm the LR, re-decay, replay — is not
implemented at all. We have the measuring tools (CL-0/1/2) and none of the
method.

**Tooling failed more than the science did.** Four kernels died behind green
uploads, and `kaggle datasets version` silently shipped a dataset with zero JSON
files. The checks that catch this are now in the suite, but they were not there
when we needed them.

**Under-powered measurements went unnoticed for a long time.** Two accuracy
thresholds were written as round numbers without ever checking what one standard
error was worth. We nearly published a noise result as a finding.

**Known gap (closed 2026-10-17):** `cosmopedia-v2`'s declared licence *inside*
`smollm-corpus` reads empty. The content's own upstream card
(`HuggingFaceTB/cosmopedia`) declares Apache-2.0, the container is tagged
ODC-By, and the project credits both — see the licence matrix's Cosmopedia
section.

---

## 5. What failed, and why it was interesting

Nine incidents are recorded in `docs/journal/`. The four that taught us
something:

**I14 — the published training log was missing its first half.** v0.1.2's HF
log began at step 1,740 because a log-restore fix landed one session after that
session had started. 347 steps were gone. Invisible to the loss curve; found by
checking that the log starts at step 10.

**I15 — training windows ignored document boundaries.** The corpus is
EOS-delimited, median document ~200 tokens against a 1,024 context, so nearly
every window crossed a boundary — and attention was unmasked. Worse, the mixer
writes four corpora in million-token round-robin blocks, so 0.3% of windows splice
two corpora with no separator. Invisible to every metric we optimise. Found by
reading the mixer.

**I16 — 41.7% of the code corpus is under non-permissive licences (39.5% GPL-family).** The licence filter existed, was
tested, and was correct. It landed **40 minutes after** the corpus that needed
it, and nothing tied a new filter to a rebuild of the datasets already built. A
green test on a filter that ran on the wrong side of a timestamp reads exactly
like safety.

**I17 — the dataset publisher was silently dropping two directories.**
`kaggle datasets version` defaults to `--dir-mode skip`. The code dataset had
been shipping with no `configs/`, no `eval/`, and zero JSON files — which is the
real reason X18's reversibility gap was never measurable. Every upload reported
success.

---

## 6. What we concluded about method

The single most useful thing we learned is not about the model:

> **Loss and accuracy are separate instruments, and optimising one does not move
> the other.**

Six architecture ablations moved loss by 0.01–0.05 nats. A corpus change moved
it by 0.2285 nats; at 2,000 samples it moved no task by more than +1.74 pp and
regressed LAMBADA by −1.25, with no 2σ accuracy movement. Twice we measured
both and got the same shape: loss improves without a proportional accuracy
gain. The problem is the data, not the capacity or the method — but the 500-
sample claim that ARC-Easy fell 3.8 points was noise.

Which is why the next version is not a new architecture. It is a **continual
update of this checkpoint** — new corpus, re-warm the learning rate, replay the
old data, re-decay, and never restart from scratch. With 30 GPU-hours a week,
that is the only way this model gets better at a rate worth having.

---

## 7. Where everything lives

| what | where |
|---|---|
| daily narrative | `docs/journal/YYYY-MM-DD.md` |
| every decision, with alternatives | `docs/DECISIONS.md` |
| incidents | `docs/journal/*.md` (`## I…`) |
| pre-registrations + hashes | `docs/preregistrations/` + `LEDGER.md` |
| verify the hashes | `python tools/verify_prereg_ledger.py` |
| literature research | `docs/research/` (5 documents) |
| licence analysis | `docs/legal/training-data-licence-matrix.md` |
| public model card | `docs/public/hf-model-card.md` |
| weights | `kalia-lm/kalia-v012` (public), `kalia-lm/kalia-v020` (private) |

**Version ladder:** v0.1.2 (first from-scratch release) → **v0.2.0 (compliance
rebuild, complete)** → v0.3.0 (first continual update, + whatever X19 earns) →
v0.4.0 (possible base rebuild, only if data selection proves the binding
constraint).
