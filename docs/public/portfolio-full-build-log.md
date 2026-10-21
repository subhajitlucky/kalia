# KALIA: a 58M language model trained from scratch on free GPUs — the build log

> अथ शब्दानुशासनम् — *"Now begins the discipline of words."*

A 58M-parameter language model trained from **random initialization** on
free-tier Kaggle GPUs (30 hours/week, 2× T4) — no pretrained weights, no
distillation, no fine-tuning, zero dollars of compute. This is the record,
organised by what was built rather than by calendar: every experiment, every
number, every decision, every mistake, including the ones that took three weeks
to surface.

**Headline results**

| | v0.1.2 | v0.2.0 |
|---|---|---|
| Params | 57,856,256 | same |
| Tokens seen | 1.82B | 2.50B |
| Held-out loss | 3.0533 / 0.9992 bpB | **2.8248 / 0.9244 bpB** |
| Zero-shot (2,000 samples) | PIQA 59.6 · ARC-E 46.0 · HellaSwag 35.6 | PIQA 61.4 · ARC-E 46.4 · HellaSwag 35.6 |
| Status | released | released as an experiment; the registered 500-sample rule fails on ARC-E and LAMBADA, but at 2,000 only LAMBADA is outside the 1.0 pp allowance (−1.25) |

- Code + engineering journal: https://github.com/subhajitlucky/kalia
- Weights: https://huggingface.co/kalia-lm/kalia-v012 · `kalia-v020`

---

## 1. The goal, and why from scratch

Fine-tuning forks someone else's brain. I wanted a model where I could account
for every byte: the corpus mixture, the tokenizer, the architecture, the
optimizer, the exact run. That costs capability — this is a storyteller, not an
assistant — and buys an auditable artifact. Every weight in KALIA exists
nowhere else.

The constraint was equally deliberate: **free compute only**. Kaggle's free tier
gives 30 GPU-hours per week on 2×T4, with background runs and persistent
notebook output. The whole system had to survive that: session caps, weekly
quota, resumable checkpoints, no secrets in API-triggered runs.

The second goal shaped everything: **make the process verifiable**. Every
experiment pre-registered before it ran, with a SHA-256 hash in a public ledger.
Every decision numbered. Every incident written down, including the
embarrassing ones — and by the end, several of them were mine, in public.

## 2. Building the machine

**Platform, by evidence.** Kaggle over Colab/Lightning: documented 30 h/week,
2×T4, background runs, 20GB persistent output, no credit card. Decided once,
never revisited.

**The codebase**, test-driven from the first green build: a decoder-only
transformer (RMSNorm, SwiGLU, RoPE, weight tying), a memory-mapped token
dataset, a resumable DDP trainer with HuggingFace checkpoint sync, a sampler, and
the Kaggle notebooks that run it.

**The data**, tokenized on free CPU where quota costs nothing: TinyStories +
FineWeb-Edu, GPT-2 BPE, uint16 shards.

**Micro-ablations before any expensive run.** Each arm: 30M params, equal tokens,
identical seed, ~30 minutes. Step-700 validation loss:

| Arm | Val loss |
|---|---|
| AdamW | 3.8041 |
| Muon | 3.5937 (−0.21) |
| Muon + QK-Norm + logit soft-cap | **3.5103** (−0.29) |

The ordering held at every checkpoint. Only then did the full-scale runs launch.
X31 (2026-10-21) later re-ran the last row as a 2×2 at two fresh seeds: the gain
is QK-Norm alone (**−0.084**); the logit soft-cap is **+0.010** and not
supported.

## 3. The first recipe, and the discipline that rejected it

**A promising result was refused.** Muon+ (one post-polar normalization step)
beat plain Muon on 7 of 7 checkpoints — by −0.015 nats, below the 0.02 promotion
threshold set in advance. It did not ship. The threshold was set first and
applied second, to our own favourite.

**The LR sweep closed the question:** 0.015 → 3.4943 · **0.02 → 3.4941** ·
0.03 → 3.5027 · 0.06 → 3.5380.

**The architecture search returned nothing.** Looped depth and grouped-query
attention were quality-neutral; thin-and-deep lost 0.2. We published the
negatives. One finding survived: the looped model ran **1.35× slower per step,
not 2×**, because reused weights stay cache-hot.

**The run told an honest story.** Validation plateaued for 1,000 steps while
training loss kept falling. The cosine decay bit at step 3,000. Then quota ran
out at step 3,478 of 4,770. A deterministic high-precision eval settled what the
noisy ones could not: **2.4366 / 0.8184 bpB** on the original held-out set — no
regression, the swings were noise.

## 4. The compliance rebuild — a licence bug found and disclosed

A per-file audit of the 5% code slice measured **41.7% of characters under
non-permissive licences, 39.5% GPL-family**. The licence filter existed, was
tested, and was correct — it landed **40 minutes after** the corpus that needed
it, and nothing tied a new filter to a rebuild of existing data. That gap had
put ~50M copyleft tokens into v0.1.2.

Two things followed, both deliberate:

- The v0.1.2 card **discloses it** rather than quietly fixing it, and the code
  slice is never released.
- "Publish all training data" was redefined as "publish the filtered corpus plus
  the filter" — redistributing copyleft text is what triggers the obligation,
  and v2b is licence-clean by construction.

The rebuilt corpus trained **v0.2.0**: 3.0533 → **2.8248** on the same yardstick,
4.6× the pre-registered bar. Its registered 500-sample stability rule failed on
ARC-Easy (−3.8) and LAMBADA (−2.2), which is why it ships as an experiment rather
than a promotion. At the 2026-10-21 2,000-sample re-measurement, the ARC-Easy
regression did not replicate (+0.45) and only LAMBADA remains outside the 1.0 pp
allowance (−1.25, z=−0.92); no movement reaches 2σ.

**The lesson that outlived the bug:** a green test on a filter that ran on the
wrong side of a timestamp reads exactly like safety.

## 5. The architecture line, and closing it

Six arms, one after another, each looking like a result and none surviving:

- **X16, reversal training** — both pre-registered bars missed on both seeds. The
  entry-exit gap got *worse*. Reversing chunk order does not teach token-level
  entry.
- **X18, gated residual** — a real 0.0436-nat gain, but the learned gate never
  opened (0.0192 against a 0.05 bar; 5e-06 of response to its input). Deleting
  the gate destroys the gain; deleting the input dependence entirely *beats* it
  (X20, −0.1358). The effect was a static per-channel modulation, not the
  data-dependent read the technique claims.
- **X20/X21** — the largest architecture effect this project ever measured,
  −0.1358 nats. Replicated at fresh seeds it **inverted to +0.0516**.

**D48 closed the line.** The reason wasn't the arms; it was that no arm had ever
been read against a measured baseline. Two independent sessions at one seed
agree to 0.0018 — but fresh seeds come in ~0.11 nats higher at 30M. Every
single-seed delta in the project was being read against a ruler nobody had
measured. They were all noise, and the ranking between them was an artefact of
comparing noise to noise.

## 6. The noise floor stops the programme

The continual-update programme (v0.3.0) opened with a measurement, not an
experiment: **how much does the 58M control vary by seed?** Three arms, two
sessions.

**0.2892 nats** — 2.5× the 30M ruler, and five times the pre-registered 0.15
stop threshold. Two sessions reproduced it to 0.0020. Steps 2–4 were cancelled
by the registration's own rule, after ~13 GPU-hours spent to learn that the
instrument couldn't resolve what the programme wanted to measure.

The pivot was to visible-gain work: inference-time levers and full-scale data
changes only.

*(Recorded later, 2026-10-18: the stop used the wrong ruler. Steps 2–4 measure
continual updates from a shared checkpoint, whose seed spread turns out to be
**0.0015–0.0070** — some 40–200× smaller. The from-scratch spread does not govern
them. The work resumed under new registrations, which was the right instinct, but
the false stop was never recorded. It is now.)*

## 7. Continual learning: three failures and a ladder

**Inference-time first.** A copy probe showed the model *can* use context: oracle
top-1 **0.828** against 0.125 chance. But BM25 top-3 — which never missed the
true passage — fell to **0.297**. The bottleneck is choosing among confusable
passages, not finding them. The log-prob selector *hurt* (−0.281); scoring only
the decision position removed the harm but never beat greedy. At 58M the model's
own probabilities do not contain the information needed to pick better
continuations. One inference-time lever did work: **entity-consistent decoding**,
+0.061 retention, 3× greedy.

**Test-time training closed null.** Across a 40× strength range, in-place fast
weights moved entity retention by +0.0000.

**The adaptive mixture ate dessert.** Reweighting sources by their own loss drifts
to the easiest data: FineWeb 60% → 22.7%, train loss down, val **+0.0177 worse**.
An improvement-aware policy halved the drift without stopping it. The failure
mode is self-reference — a policy reading a signal the model also optimises.

**The replay ladder.** Four Vedic recitation orderings as replay schedules gave a
monotone val staircase (uniform < krama < jata < ghana, ~0.02-nat rungs) and a
retention ladder in the same order. Structure costs, in proportion to its depth.

## 8. The audit: the record against itself

An external audit of everything published, then a whole-record audit — six
parallel reads of 72,000 lines of journal, decisions, registrations, results,
research and public documents.

The finding was not about the model. **The project's distinctive asset is its
record, and the record was the least-verified thing it owned.** Nine published
claims contradicted the internal evidence: a model card called a *training-stream
prefix* "held-out data"; the README said the model wasn't public while linking to
it; the README had one experiment's conclusion **backwards**; a licence matrix
denied shipping data that sits two rows above it in the same table.

And a structural pattern across 24 incidents: **every root cause with a named
tool attached is closed; every one whose fix is "be careful" is open.** Stale code
bundles (7), self-referential measurement (5), thresholds below the metric's own
noise (7), claims outliving their evidence (7), checks that never ran (8), silent
no-ops (7). One safety tool was invoked by nothing but its own documentation.

**What was built in response** — the part worth showing:

- a claims checker that verifies the public documents still agree with the
  evidence, and fails CI when they don't, with every rule named after the defect
  it prevents;
- the decorative checks wired into CI, so convention became enforcement;
- a **run manifest** on every training job: seed, code fingerprint, every input's
  sha256 — so a run proves what it was instead of being reconstructed by hand;
- verdicts **computed from the archived logs**, not transcribed.

## 9. Two reversals, one pattern

The day's two headline results both dissolved on inspection, and both for the
same reason.

**Replay.** The claim was that 10% replay + rewarm beats no-replay. With a clean
control — same runner, same corpus, same budget, replay the only variable — the
0.088–0.094 nat gap decomposed into **6% replay and 94% delivery path** (one arm
sampled 498M tokens ~1.05 times, the other 2.4B ~0.2 times). Replay's own effect
is a small, reproducible *penalty*.

**Document masking.** A registered correctness fix, never run, with every shipped
model trained across document boundaries. It measured **+0.0444 worse** — but
both arms were *evaluated* without the mask, so the fixed model was tested in a
regime it had never trained in. The registration had predicted exactly this, in
its pass branch. Scoring all four combinations:

| | eval unmasked | eval masked |
|---|---|---|
| train unmasked | 4.8037 | 4.8122 |
| train masked | 4.8484 | **4.7998** |

The penalty was **100% the mismatch**. Matched, masking is *better* by 0.0124,
at both seeds.

**The pattern, twice in one day:** the thing everyone believed was caused by the
thing being studied, and both times it was caused by something else that wasn't
held steady. First the book, not the revision. Then the exam, not the fix.

## 10. Every decision (D1–D54)

Selected, with the full ledger in `docs/DECISIONS.md`.

| # | Decision | Outcome |
|---|---|---|
| D1–D8 | Platform, from-scratch, scale, corpus, ladder, micro-ablations, pre-registration, journal | held |
| D15 | Muon+ below threshold → not promoted | enforced |
| D21 | Distillation rejected | closed |
| D25 | RL self-play and MoE rejected at this scale | closed |
| D41 | Stop v0.1.2 at step 3,478 | done |
| D42 | Publish negative results as prominently as wins | **active — the project's spine** |
| D43 | The corpus rebuild redefined the yardstick; re-baseline both sides | active |
| D44 | Copyleft disclosure + "publish the filter, not the raw corpus" | active |
| D45 | Thresholds must exceed the measured noise floor | active |
| D46 | The gated-residual gain is real but is not a gate | closed |
| D47 | X20 static gate: best effect ever, nothing promoted without replication | done |
| D48 | **Architecture line closed; no more single-seed arms** | active |
| D49 | A load-bearing citation was about the wrong paper; retracted | closed |
| D50 | The replay default's justification was unlocatable; removed | closed |
| D51 | `--resume` silently continued the source schedule | fixed |
| D52 | The v0.2.0 card denied a redistribution the repo ships | fixed |
| D53 | WSD lands as mechanism; the decay shape stays a knob | active |
| D54 | The record is the product, so the record is now checked | active |

## 11. Every incident (I1–I24)

Full table in `docs/journal/`. The ones worth reading:

| # | Incident | Root cause → fix |
|---|---|---|
| **I6** | Resume crash (`EOFError`) | Both DDP ranks wrote the same checkpoint → single downloader, atomic swap, barrier |
| **I7** | Latent crash on older checkpoints | Optimizer `load_state_dict` dropped new keys, silently → defaults restored + regression test |
| **I14** | Published log began at step 1,740 | Log-restore fix landed mid-run → resume restores logs; verify the first step |
| **I15** | Training windows ignored document boundaries | No notion of a "document" anywhere → `document_mask()`, and X17 |
| **I16** | 41.7% of the code corpus is copyleft | The filter landed 40 min after the corpus → disclosure, rebuild, rebuild rule |
| **I17** | The dataset publisher dropped two directories | `--dir-mode skip` → round-trip verify |
| **I18–I20** | Continual arms would train zero steps; the control re-warmed too | `update_budget`, gated on `rewarm` → three tests |
| **I22** | Six runs died with blank errors | A notebook-format warning became fatal → cell ids, static check |
| **I23** | Stale code bundle loaded from a wrong mount | Fallback to *any* `train.py` → assert files, prefer the code dataset |
| **I24** | A contamination gate reported 63,491,977 hits | Every one was whitespace → `MIN_DISTINCT` |

## 12. The mistakes that stayed in

- **Muon+ not promoted** despite winning 7/7 checkpoints — threshold first, decision second.
- **The architecture search returned nothing**, and the one large effect inverted under replication.
- **A documentation error caught late**: our docs described v0.1.2 as "Muon+" for hours; the config proved otherwise. Corrected, and the hashed pre-registration got a registered amendment rather than a silent edit.
- **The plateau**, documented instead of massaged, then resolved by a deterministic eval.
- **A programme stopped on the wrong ruler** (0.2892 from-scratch applied to 0.0015 continual), unrecorded for ten days.
- **Nine claims that drifted from the evidence** — caught by an external audit, then a whole-record one, then made impossible by a checker.

## 13. What's next

- **The memory-only rung**: retrieval as a *tool* plus entity-consistent decoding —
  the one inference-time lever that measured a win, at zero GPU cost.
- **Weight updates from use** must beat a **zero-GPU-hour baseline** before they
  run again. The scoreboard so far: data 1, inference-time 1, weight updates 0.
- **Pin the code bundle** rather than fingerprint it — the largest remaining
  unengineered failure class.
- **The benchmark sample size** raised from 500 to 2,000 for any future accuracy claim.

## 14. Reproduce it

```bash
git clone https://github.com/subhajitlucky/kalia && cd kalia
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python -m pytest -q                      # 522 tests
python tools/audit_claims.py --check     # public documents still match the record
python tools/verify_prereg_ledger.py     # every registered hash still verifies
```

Data prep, training and evaluation run on free Kaggle notebooks; `notebooks/`
contains every one. Pre-registration hashes are in
`docs/preregistrations/LEDGER.md`; every run's raw logs are in `docs/results/`.

*KALIA is a personal research project. Not affiliated with any government
scheme, company, or other project using a similar name.*
