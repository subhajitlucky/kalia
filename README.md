# KALIA

> अथ शब्दानुशासनम्
>
> *"Now begins the discipline of words."* — the opening invocation of Pāṇini's
> Aṣṭādhyāyī, the oldest surviving systematic textbook of language.

A from-scratch ~58M-parameter language model — trained on free Kaggle GPUs with zero
fine-tuning, zero pretrained weights, and zero budget.

| release | status | tokens | held-out loss | notes |
|---|---|---|---|---|
| [`kalia-v012`](https://huggingface.co/kalia-lm/kalia-v012) | released | 1.82B | 3.0533 | stopped at 73% of its schedule |
| `kalia-v020` | experiment, not yet public | 2.50B | **2.8248** | full 4,770-step schedule, compliance-rebuilt corpus |

Both figures are on the **same** held-out set (the compliance-rebuilt `kalia-prep-v2b`),
scored by the same deterministic 100-batch protocol. The 2.4366 that appeared in earlier
releases is v0.1.2 on the *original* held-out set, which the rebuild replaced; the two are
never compared (decision D43).

**The complete record — method, results, strong points, weak points, and every failure —
is in [`docs/SYNTHESIS.md`](docs/SYNTHESIS.md).** 45 decisions, 9 incidents, 9
hash-verified pre-registrations, 213 tests, machine-readable run results, verified citations.

Every weight in KALIA is learned by this project's own training run; nothing is borrowed
from another model.

## Spec

| Field | Value |
|---|---|
| Parameters | 57,856,256 |
| Layers / heads | 10 / 8 |
| Embedding dim | 512 |
| Context length | 1,024 tokens |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Architecture | Pre-norm decoder-only: RMSNorm, SwiGLU, RoPE, QK-Norm, logit soft-cap (τ=30), weight tying, no biases |
| Optimizer | Muon (Newton–Schulz) on hidden matrices; AdamW for embeddings/head/norms |
| Schedule | Cosine with warmup, 4,770 steps (~0.5M tokens/step) |
| Training hardware | Kaggle free tier: 2× NVIDIA T4, ~30 GPU-hours/week |
| Training data | 2.4B tokens, 60/20/15/5 — FineWeb-Edu (ODC-By-1.0) / TinyStories (CDLA-Sharing-1.0) / Cosmopedia v2 (Apache-2.0) / permissively-licensed Python |

Full per-source licensing analysis, including what may and may not be redistributed, is in
[`docs/legal/training-data-licence-matrix.md`](docs/legal/training-data-licence-matrix.md).

## Results

**Held-out loss, v0.1.2 → v0.2.0** — same weights protocol, same held-out set, only the
corpus changed: **3.0533 → 2.8248, a 0.2285-nat improvement**, 4.6× the pre-registered
0.05-nat bar. 0.9992 → **0.9244** bits-per-byte.

**Zero-shot accuracy** (lm-eval 0.4.9, 0-shot, 500 samples). Standard errors at this sample
size are ~2 points, so the deltas are read with that attached:

| task | v0.1.2 | v0.2.0 | Δ | σ |
|---|---|---|---|---|
| PIQA | 61.4 | 63.8 | +2.40 | 1.1 |
| HellaSwag | 36.8 | 39.8 | +3.00 | 1.5 |
| WinoGrande | 50.2 | 50.8 | +0.60 | inside noise |
| LAMBADA | 23.0 | 20.8 | **−2.20** | 1.2 |
| ARC-Easy | 45.8 | 42.0 | **−3.80** | 1.7 |

**Three of the five movements are smaller than the measurement's own noise and cannot be
called improvements.** The pre-registered stability rule allows no task to regress by more
than 1.0 point, and v0.2.0 **fails it on two** — which is why it is held as an experiment
rather than promoted over v0.1.2. The thresholds were not moved after seeing the result,
and the noise floor is published beside the verdict rather than used to rescue it (D45).

**What the loss/accuracy split means.** Changing the data moved loss by 0.2285 and
still cost 3.8 points on ARC-Easy. When loss improves and accuracy does not follow,
the problem is the data, not the capacity or the method. That is the project's main
methodological finding and it is what the next version is built on.

**The architecture ablations produced nothing (D48).** Seven micro-arms were run on a
30M budget. The best of them — a static learned per-channel modulation, X20 — looked
like the project's largest result ever at −0.1358 nats, and beat a data-dependent gate
that actually reads its input. It was replicated at two fresh seeds and **inverted to
+0.0516, worse than control** (X21, R-1/R-2/R-4 all fail).

The replication also produced the number the project never had: **the control's own
spread**. Two independent sessions at seed 1337 agree to 0.0018, so the machine is not
the variable — but fresh seeds 1338/1339 come in ~0.11 nats higher. That baseline spread
**exceeds X18's entire −0.0436 "effect"** and X19's −0.0018 by an order of magnitude.
Every single-seed delta in this project was read against a baseline that was never
measured. They were all noise, and the ranking between them was an artefact of
comparing noise to noise. **No architecture arm may be run from one seed again.**

**Micro-ablations** (30M params, equal tokens, equal seed, step-700 val loss):

| Arm | Val loss |
|---|---|
| AdamW | 3.8041 |
| Muon | 3.5937 |
| Muon + QK-Norm + soft-cap | **3.5103** |

Muon beat AdamW by 0.21 nats; QK-Norm plus soft-cap a further 0.08. LR 0.02 was optimal
across a 4-point sweep. At full scale the Muon model passed the AdamW baseline's *final*
loss using ~23% fewer tokens (3.2214 @ step 1730 vs 3.2702 @ step 2250).

## What failed, and what it taught

Negative results are published with the same prominence as wins. That is decision D42, and
it is the reason this repository is worth reading.

| What | Result | What it taught |
|---|---|---|
| Reversal training (X16) | **Rejected.** Both pre-registered bars missed on both seeds — the entry-exit gap got *worse* | Reversing chunk order does not teach token-level entry. The gap is architectural, confirmed at 5.1 nats at 30M, not an undertraining artifact |
| Gated residual (X18) | **Not attributable.** Best loss gain of any arm (4.4× bar) but the learned gate never opened — 0.0192 against a 0.05 bar, and only 5e-06 of variation in response to its input | The measured effect belonged to the branch normalisation, not to the data-dependent read the technique claims. Its documented init was also dead code, silently overwritten by the global re-init |
| Document masking (X17) | Implemented and pre-registered, not yet run | — |
| Licence audit (I16) | **41.7% of the code corpus is copyleft.** The filter existed and was tested — it landed 40 minutes *after* the corpus that needed it | A green test on a filter that ran on the wrong side of a timestamp reads exactly like safety |
| Dataset publishing (I17) | The Kaggle publisher defaults to `--dir-mode skip`, so the code dataset shipped with no configs, no eval folder, and no JSON at all | Every upload reported success. Assert the interface, then trust it |

**Two measurement mistakes were our own.** A corpus rebuild silently replaced the
held-out set, making a healthy run look 0.74 nats worse (D43). And two pre-registered
accuracy thresholds — 0.5 pp and 1.0 pp — sat *below* the ~2 pp standard error of the
benchmarks they were read from, so one of them passed on noise alone (D45).

## Repository layout

```
kalia/
  model.py               # RMSNorm, SwiGLU, RoPE, attention, QK-Norm, doc masking, GPT
  data.py                # memory-mapped token dataset, document masks, replay mixture
  mixture.py             # per-source sampling + held-out per-source loss (Kautilya arm)
  prepare.py             # tokenize text sources into uint16 .bin shards (licence filter)
  prep_longform.py       # document-length probe per source
  mix_bins.py            # blend shards into a training mixture
  train.py               # resumable, time-budgeted, DDP training loop
  ablate.py              # sequential micro-ablation runner (+ Abhimanyu-gap readout)
  optim.py               # Muon / Muon+ / AdamW hybrid optimizer
  sample.py              # generate text from a checkpoint
  eval_bench.py          # zero-shot suite via lm-evaluation-harness
  eval_val.py            # deterministic held-out loss (the registered primary metric)
  eval_reversibility.py  # entry-exit asymmetry (Abhimanyu gap)
  eval_probes.py         # frozen 27-prompt probe suite
  gate_probe.py          # Gated Residual gate statistics (mean and dispersion)
  lambada_loader.py      # serve LAMBADA to lm-eval without its dead loading script
  forgetting_probe.py    # continual-learning retention ledger
  audit_licences.py      # per-file licence audit
  tools/                 # dataset publisher, model exporter, figure + notebook builders
  configs/               # the real model config + micro-* ablation arms
  notebooks/             # Kaggle kernels: prep, train, ablate, benchmark, evaluate
  docs/results/         # experiments.json + archived raw logs: every run, machine-readable
  test_*.py              # 213 tests, flat at the repo root
  docs/SYNTHESIS.md      # the whole record in one document
  docs/journal/          # dated engineering journal
  docs/DECISIONS.md      # 45 numbered decisions
  docs/research/         # technique surveys with honest prior-art verdicts
  docs/preregistrations/ # hash-anchored pre-registrations + LEDGER
  docs/legal/            # training-data licence matrix
  docs/public/           # model cards, publish checklist
```

## Local quickstart

Heavy work belongs on Kaggle. The laptop runs the test suite and nothing else.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # 213 tests, ~22s on CPU
python tools/check_notebooks.py   # static-checks every notebook before spending quota
python tools/collect_results.py --check   # results data still matches its logs
```

Two rules this repository learned the hard way, both enforced by tools rather than by
convention:

- **Never download large artifacts to the developer machine.** Measure on a Kaggle CPU
  kernel, which costs neither GPU quota nor bandwidth. `tools/publish_code_dataset.py`
  and `tools/publish_model_card.py` verify by round-trip for the same reason: a remote
  write that reports success is not evidence the write landed.
- **Never add a commit trailer.** This repo's history carries none. See
  [`docs/COMMIT_CONVENTION.md`](docs/COMMIT_CONVENTION.md).

## Kaggle runbook

1. `notebooks/kalia-prep*.ipynb` (CPU) → `Save Version` → `Output` → `Create Dataset`
   (private). This is the tokenised corpus.
2. `notebooks/kalia-train-v020.ipynb` (GPU T4 ×2, Internet on, `HF_TOKEN` secret) →
   every later run resumes from the latest checkpoint on the Hub. Nothing is ever lost.
3. `notebooks/kalia-v020-final-eval.ipynb` (CPU) → the registered final evaluation.
4. `tools/publish_model_card.py` → the model card, verified by downloading it back.

`train.py` checkpoints to the Hub every 30 minutes and at session end, so the weekly quota
caps a session but never the run.

## Version history

- **v0.1.0** — baseline: AdamW, RoPE, SwiGLU, RMSNorm, fp16, T4×2. Final val 3.2702.
- **v0.1.1** — Muon for hidden weight matrices. Micro-ablation at equal tokens:
  3.5937 vs 3.8041, a 0.21-nat win.
- **v0.1.2** — QK-Norm + logit soft-cap (τ=30). Micro-ablation 3.5103, 0.29 vs baseline.
  Stopped at step 3,478 of 4,770 on quota with the plateau documented. First public release.
- **v0.2.0** — compliance-rebuilt corpus, same recipe, full 4,770 steps. Val 2.8248
  (−0.2285 nats vs v0.1.2 on the same yardstick); three of five benchmarks move inside
  their own noise, two regress beyond it. Held as an experiment: it fails its own
  pre-registered stability rule on ARC-Easy and LAMBADA.
- **v0.3.0** — *not* a new base. The first **continual update** of v0.2.0: new corpus, the
  learning rate re-warmed and re-decayed, old data replayed, and no restart from scratch.
  With 30 GPU-hours a week, that is the only way this model improves at a rate worth
  having — and the replay ratio has to be measured at 58M rather than inherited, because
  published replay results stop at 0.6B backbones.
- **v0.4.0** — possible base rebuild, only if instance-level data selection proves the
  mixture is the binding constraint.

## License

Code: MIT (see `LICENSE`). Model weights: Apache-2.0.

Training data is **not** redistributed. The v0.2.0 corpus is licence-clean by
construction and could be released under three attribution conditions; the recipe,
manifests and filter ship instead, which reproduces it exactly, since every source is
public. The v0.1.2 code slice is **never** released — it was built without a licence
check and is 41.7% copyleft. That is disclosed on the v0.1.2 card rather than quietly
fixed.
