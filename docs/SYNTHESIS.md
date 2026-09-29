# KALIA — the complete record

One document that ties the rest together. Everything here links to its primary
source; nothing is asserted here that isn't measured somewhere in this repo.

**What this is:** a 57.9M-parameter language model trained from scratch, on free
Kaggle GPU time, with every decision, failure and negative result written down as
it happened. Seven days, 124 commits, 45 decisions, 9 incidents, 9 hash-verified
pre-registrations, 137 tests.

---

## 1. What we actually built

**Architecture** — 57,856,256 parameters, decoder-only, pre-norm.

| component | choice | why |
|---|---|---|
| optimizer | Muon on hidden weights, AdamW on embeddings/scalars | micro-ablation: −0.21 nats; Kimi K2 and GLM-4.5 use it |
| attention | QK-Norm + logit soft-cap (τ=30) | micro-ablation: a further −0.08 nats |
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

**Zero-shot benchmarks** (lm-eval, 0-shot, 500 samples):

| task | v0.1.2 | v0.2.0 | Δ | S-A (≤1.0pp) |
|---|---|---|---|---|
| PIQA | 61.4 | 63.8 | **+2.40** | ok |
| HellaSwag | 36.8 | 39.8 | **+3.00** | ok |
| WinoGrande | 50.2 | 50.8 | +0.60 | ok |
| ARC-Easy | 45.8 | 42.0 | **−3.80** | **FAIL** |
| LAMBADA | 23.0 | *pending* | — | never ran |

**Read the noise before the numbers.** Standard errors at `limit=500` are
1.99–2.24 pp, so PIQA, HellaSwag and WinoGrande are *not* separable from
v0.1.2, and ARC-Easy's −3.80 is ~1.7σ. The loss improvement is the solid
result; the benchmark table is directional.

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
copyleft, found in 11 minutes of CPU, and disclosed on the live model card rather
than quietly fixed.

**Reproducibility is real.** 137 tests, a resumable multi-session harness, a
dataset publisher that verifies by round-trip, and a ledger whose hashes anyone
can check.

---

## 4. Weak points — stated plainly

**One benchmark genuinely regressed.** ARC-Easy −3.80 against a 1.0 pp
allowance. The registered rule is symmetric with the success criterion and is
reported as it stands. A 5%→10% code share costing 3.8 points on ARC-Easy says
the mixture is **far more sensitive than we have been treating it**, and that is
the most important open question in the project.

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

**Known gap:** `cosmopedia-v2`'s declared licence *inside* `smollm-corpus` reads
empty. The last unverified cell in the licence matrix.

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

**I16 — 41.7% of the code corpus is copyleft.** The licence filter existed, was
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
it by 0.2285 nats — and cost 3.8 points on ARC-Easy. Twice we measured both and
got the same shape. When loss improves and accuracy does not follow, the problem
is the data, not the capacity or the method.

Which is why the next version is not a new architecture. It is a **continual
update of this checkpoint** — new corpus, re-warm the learning rate, replay the
old data, re-decay, and never restart from scratch. With 30 GPU-hours a week,
that is the only way this model gets better at a rate worth having.

---

## 7. Where everything lives

| what | where |
|---|---|
| daily narrative | `docs/journal/YYYY-MM-DD.md` (4 entries) |
| every decision, with alternatives | `docs/DECISIONS.md` (45) |
| incidents | `docs/journal/*.md` (`## I…`) (9) |
| pre-registrations + hashes | `docs/preregistrations/` + `LEDGER.md` (9) |
| verify the hashes | `python tools/verify_prereg_ledger.py` |
| literature research | `docs/research/` (5 documents) |
| licence analysis | `docs/legal/training-data-licence-matrix.md` |
| public model card | `docs/public/hf-model-card.md` |
| weights | `kalia-lm/kalia-v012` (public), `kalia-lm/kalia-v020` (private) |

**Version ladder:** v0.1.2 (first from-scratch release) → **v0.2.0 (compliance
rebuild, complete)** → v0.3.0 (first continual update, + whatever X19 earns) →
v0.4.0 (possible base rebuild, only if data selection proves the binding
constraint).
