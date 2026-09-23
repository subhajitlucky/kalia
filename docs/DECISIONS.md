# KALIA Decision Ledger

Every significant decision: what, alternatives, evidence, status. Newest last.

| ID | Date | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|---|
| D1 | 2026-09-22 | Train on Kaggle free tier | Colab, Lightning AI, Modal | 30 GPU-h/week documented, T4×2 32GB, background runs, no card | active |
| D2 | 2026-09-22 | From-scratch training (no fine-tuning) | Fine-tune a small base model | Ownership requirement; weights exist only here | active |
| D3 | 2026-09-22 | 58M-param decoder-only transformer | 25M / 125M | Compute budget vs capability trade-off | active |
| D4 | 2026-09-22 | GPT-2 BPE tokenizer (off-the-shelf) | Train custom 32k BPE | Avoids tokenizer-training stage; adequate for English | active |
| D5 | 2026-09-22 | TinyStories + FineWeb-Edu mixture | TinyStories only; FineWeb only | Fluency + facts; TinyStories proven at tiny scale | active |
| D6 | 2026-09-22 | Resumable multi-session harness with HF checkpoints | Single long run | Free tier sessions are capped; resumability removes waste | active |
| D7 | 2026-09-22 | Adopt Muon for hidden weights | AdamW only | Micro-ablation: −0.21 val loss; Kimi K2, GLM-4.5 use it | validated |
| D8 | 2026-09-22 | Adopt QK-Norm + logit soft-cap | Soft-cap only; QK-Norm only | Micro-ablation: additional −0.08; GLM/speedrun practice | validated |
| D9 | 2026-09-22 | Batch 8 × grad-accum 32 instead of 32 × 8 | Reduce context to 512; gradient checkpointing | OOM fix without changing effective batch size | active |
| D10 | 2026-09-22 | Only browser-started Kaggle runs for training | API-triggered runs | API runs cannot read secrets (I1); fail-fast HF test added | active |
| D11 | 2026-09-23 | Freeze v0.1.0 after its first session | Continue to full 4770 steps | Its curve is the equal-token baseline; quota is better spent on the winning recipe | active |
| D12 | 2026-09-23 | Incremental ladder with micro-ablations before full runs | Train best-guess recipe directly | One change per version keeps attribution; 30M/50M-token screens cost ~30 min | active |
| D13 | 2026-09-23 | Journal everything; 3-source research protocol | Ad-hoc decisions | Requested; also how real labs operate | active |
| D14 | 2026-09-23 | Implement Muon+ as the next experiment (E1) | NorMuon; Polar Express; skip straight to architecture | Muon+ wins 60M–7B comparisons, zero extra optimizer state, one-line change | done — validated |
| D15 | 2026-09-23 | E1 result (−0.015 nats, 7/7 checkpoints) is below our 0.02 promotion threshold: keep as optional flag, queue E1b (LR interaction) instead of auto-promoting | Promote anyway; discard; re-run with more seeds | Protocol discipline: threshold exists to prevent wishful promotion; effect matches published magnitude but needs the LR lever tested | active |
| D16 | 2026-09-23 | Fix resume race with single-downloader + atomic swap + DDP barrier; restore missing optimizer keys on load | Retry loops; rank-local files | Deterministic, testable; regression tests added (I6, I7) | done |
| D17 | 2026-09-23 | Adopt expert reprioritization: LR sweeps before tricks; no full runs before recipe lock; bpB + shuffled val + 2 seeds; data mix before architecture; target-loss stopping | Continue current ladder ordering | Efficiency with limited free compute; see docs/plans/2026-09-23-expert-reprioritization.md | active |
| D18 | 2026-09-23 | Finish v0.1.2 (P5) instead of abandoning at 71%; prepare v0.2.0 in parallel (P3/P4, zero GPU) | Kill v0.1.2 now and start v0.2.0 | Complete model this week; prep work costs no GPU quota | active |
| D19 | 2026-09-23 | Target data mixture v2: 55% FineWeb-Edu / 20% TinyStories / 15% Cosmopedia / 10% code+math | Current 19/81 TinyStories/FineWeb | Data quality/ceiling beats architecture tricks at this scale | planned |
