# Strategic Audit — End of Day 2

Status: active plan
Date: 2026-09-23
Scope: everything built, measured, queued, and wasted since project start.

## 1. What exists

| Layer | State |
|---|---|
| Code | `model.py` (RMSNorm, RoPE, SwiGLU, attention, QK-Norm, soft-cap, GQA, looping), `optim.py` (Muon, Muon+), `train.py` (DDP, AMP, resume, target-stop, EMA, bpB), `prepare.py` + `mix_bins.py` (5 sources, license filter), `sample.py`, `ablate.py`, 4 evaluation modules, **60 tests** |
| Data | v0.1.x corpus (TinyStories + FineWeb-Edu, 2.46B tokens); **compliance-clean v2b** (2.4B: 60% FineWeb-Edu-dedup / 20% TinyStories / 15% Cosmopedia / 5% permissive Python) |
| Infra | 6 Kaggle kernels, 2 datasets, 2 private HF repos; CPU preps are quota-free |
| Record | 38 decisions, 11 incidents, 11 research documents, 1 hash-anchored pre-registration |
| Models | v0.1.0 baseline (final 3.2702 @ 2,250 steps); v0.1.2 in training (3.10 train @ 2,660 steps; val plateau 2.53–2.63 over 1,000 steps) |

## 2. Measured results (the only claims we can post)

- Micro-ablation: AdamW 3.8041 → Muon 3.5937 → Muon+QK-Norm+soft-cap **3.5103**.
- Muon+ vs Muon at matched tokens: 3.4941 vs 3.5091 (7/7 checkpoints).
- LR sweep: 0.015: 3.4943 · **0.02: 3.4941** · 0.03: 3.5027 · 0.06: 3.5380.
- Full-scale: baseline final 3.2702 @ 2,250; v0.1.2 reached 3.2214 @ 1,738 (~23% fewer tokens).
- Looping economics: 29.92M params identical to control; ~19.2k tok/s vs ~26k (≈1.35×
  slower, not 2×) — weight reuse keeps the looped passes cache-hot.
- First coherent generations at loss 3.22 (`docs/samples/`).

## 3. Mistakes and waste (honest)

1. **v0.1.0 full baseline (8.5 GPU-h)** — defensible as a baseline artifact, but an
   expert would have relied on micro-ablations alone (D17 acknowledges this).
2. **Micro runs wrote 1GB checkpoints every 20 min** — pure waste (I11), now fixed.
3. **Elapsed-time misreport** — an incident report claimed >5h for a 44-minute run;
   corrected using kernel run-start times (I11).
4. **Queue inflation** — ~26 experiment IDs written, 6 run. Research output is
   far ahead of execution; this is the main efficiency risk.
5. **Val plateau** — v0.1.2's validation loss has not improved in 1,000 steps
   while training loss fell; the final cosine decay may or may not fix it.

## 4. Quota accounting

At v0.1.2 session-3 start the account had **24 GPU-hours** remaining. Consumed
since: LR sweep (~1.5h) + architecture ablation (~2.5h to completion) +
session-3 remainder (~3.5h) → **≈16h left** after current jobs.

## 5. Do — ranked

1. **Finish the architecture ablation** (running; verdict ≈ 11:30 UTC) and record
   the winner. This is the last open architecture variable.
2. **Run v0.1.2 session 4 to completion** (~6.5h, within one session) with
   `--target-val-loss 2.45 --decay-steps-after-target 400`. Rationale: the
   cosine decay tail is where final quality arrives; stopping before it makes
   the release look abandoned rather than designed.
3. **Run X16 (Chakravyuha reversal), pre-registered, concurrently** (~1.5h) —
   the single postable experiment: it has a metric (Abhimanyu gap), a hash, and
   a 2×-effective-token claim to test.
4. **Run the evaluation suite on the final checkpoint** (zero GPU): probe
   report, Abhimanyu gap, entity consistency, baseline-vs-v0.1.2 comparison.
5. **Publish the milestone**: GitHub public, HF model card + weights, Kaggle
   notebooks, portfolio article — all with prior-art citations and the plateau
   documented as a finding.
6. **After quota reset: v0.2.0** = v2b data + architecture winner + X16 if it won
   + WSD/EMA/target-stop, then a second publication.

## 6. Do not

1. **Freeze the experiment queue.** No new families before v0.2.0 (only X16, and
   X15 Apoha-ELECTRA if quota allows).
2. **No more ancient-text audits until the first post is live** — auditing costs
   no GPU but expands the queue; the record already holds five traditions.
3. **No MoE, KDA, DSA, distillation, or RL self-play** (D21/D25 stand on evidence).
4. **No further LR or hyperparameter sweeps** (done; D31).
5. **Never publish data, chat transcripts, or uncleared code** (D27 checklist).
6. **No claims without: citation of closest prior work + measured number +
   config/seed + (where applicable) a pre-registration hash.**
7. **Do not start v0.2.0 before its recipe is frozen** (architecture verdict +
   X16 result in hand).

## 7. The one open decision

Finish v0.1.2 (option 2 above) or stop it at session-3 end and bank the ~7h for
v0.2.0. Recommendation: **finish it** — a designed stopping point plus a decayed
LR produces the strongest publishable artifact, and v0.2.0 deserves a clean
start on the next quota cycle.
