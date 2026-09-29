# KALIA Engineering Journal

Every decision, result and incident is recorded here. No silent changes.

## Protocol

1. **Daily entries** in `docs/journal/YYYY-MM-DD.md`: what was decided, what was
   run, what happened, with raw numbers.
2. **Decision ledger** in `docs/DECISIONS.md`: every significant decision with
   alternatives considered and the evidence behind the choice.
3. **Research backlog** in `docs/research/`: candidate techniques from the
   literature, each with sources, expected gain, risk and experiment status.
4. **Rules**: numbers beat vibes; failures get recorded with the same rigor as
   wins; every experiment names its config, seed, and artifact location.
5. **Research protocol for decisions**: read at least three independent sources,
   write the trade-off down, then decide. If sources conflict, run our own
   micro-ablation.

## Index

| Date | File | Highlights |
|---|---|---|
| 2026-09-22 | `2026-09-22.md` | Build day: design, 13 tasks, TDD, incidents, ablations, two full runs launched |
| 2026-09-23 | `2026-09-23.md` | First coherent generations, overtake of baseline, RSI plan, technique backlog |
| 2026-09-27 | `2026-09-27.md` | X16 rejected (both pre-registered bars missed on both seeds), v0.1.2's training log recovered from hub history after a mid-run code change dropped it (I14), the rebuilt val set turns out to be a different yardstick (D43), v0.2.0 session 1 |
| 2026-09-28 | why-27b-beats-397b | `docs/research/2026-09-28-why-27b-beats-397b.md` | frontier gain decomposition; data >> architecture; test-time compute is the transferable 1000x lever; D1-D8 queued |
| 2026-09-28 | open-source-landscape | `docs/research/2026-09-28-open-source-landscape-and-transfers.md` | 30 orgs, last two generations each; four unconditional transfers survive the cost/yardstick/D2/evidence filters |
| 2026-09-28 | sanskrit-reading-list | `docs/research/2026-09-28-sanskrit-reading-list-50.md` | 50 works audited with honest verdicts; Chakravala/Pingala/Katapayadi are the real algorithms; corpus is silent on learning mechanisms |
| 2026-09-28 | journal | `2026-09-28.md` | 41.7% of the code corpus measured copyleft (I16) and disclosed on the live card (D44); v0.2.0 finishes the full 4,770-step schedule and plateaus for its last 31%; X18 closed with the gate measured inert in both moments (G-0); our own accuracy thresholds found to sit below the benchmark's own noise (D45); X19 registered with an amendment correcting an error in it; CDLA-Sharing read, so the corpus is redistributable |
| 2026-09-28 | X17-doc-mask | `docs/preregistrations/2026-09-28-X17-doc-mask.md` | intra-document masking micro-ablation: 2 arms, 30M, 500 steps, bar 0.010 nats (same as D15) |
| 2026-09-28 | X18-frontier | `docs/preregistrations/2026-09-28-X18-frontier-attention.md` | NoPE and Gated Residual: NoPE rejected, gated -0.0436 nats but the gate measured inert; F-3's accuracy half found to be 0.54 sigma |
| 2026-09-28 | X19-branch-norm | `docs/preregistrations/2026-09-28-X19-branch-norm-isolation.md` | isolates the branch normalisation from the dead gate; one arm, G-1 bar 4.7544 |
| 2026-09-28 | X19-am1 | `docs/preregistrations/2026-09-28-X19-amendment-1.md` | corrects G-2's parameter arithmetic, which was wrong in the direction that flattered the arm |
| 2026-09-28 | legal | `docs/legal/training-data-licence-matrix.md` | per-source licence analysis; all four sources permit weight release, only v0.2.0's corpus is redistributable |

