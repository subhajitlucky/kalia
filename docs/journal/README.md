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

> Looking for the whole thing in one place? `docs/SYNTHESIS.md` — what we
> built, the results, strong points, weak points, and what failed and why.

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
| 2026-10-05 | `2026-10-05.md` | v0.3.0 pre-launch: amendment 1; launch-blocker I18 (continual arms never resumed) plus I19/I20 caught pre-GPU; X25 closed as a null across a 40× strength range; corpus rebuild moved local (I21) |
| 2026-10-08 | `2026-10-08.md` | 58M control spread 0.2892 (2.5× the 30M ruler) trips the registered programme-stop for Steps 2–4; corpus contamination gate; Tier-1 retrieval/best-of-N probes; publish pipeline completes end to end |
| 2026-10-09 | `2026-10-09.md` | X26 pilot: the level policy eats dessert (FineWeb 60→22.7%, train 2.63→1.79, val +0.0177); X27 RCAA withdrawn pre-GPU — the design leaked the future token by construction |
| 2026-10-11 | `2026-10-11.md` | X28: static and level replicate exactly (≤0.003 per checkpoint), dessert cost +0.0183, trend halves the drift; entity-decode triples retention (+0.061 over greedy) |
| 2026-10-12 | `2026-10-12.md` | X29 arms 1–2: uniform < jata on val; second CPU-invisible device-contract bug found on GPU; retention probe self-check refuses the wrong instrument (+0.1107); X30 drafted |
| 2026-10-13 | `2026-10-13.md` | X29 closed: monotone four-rung staircase on val and retention (krama −0.0007 completes it); registered verdict pending the second seed |
| 2026-10-17 | `2026-10-17.md` | External artifact + token-exposure audit (nothing exposed); corpus disclosure shipped (D52); tooling hardening (ledger drift classes in CI, curve-parser fix, decay clamp); seed-1402 replicas launched |
| 2026-10-18 | `2026-10-18.md` | The second seed lands and the two instruments disagree: val favours 10% replay (+0.0125/+0.0073), CL-0 favours no-replay (+0.0884/+0.0939, 22× the seed spread); the probe kernel could not name the no-replay arm, so the static arm is scored on CL-0 here for the first time at either seed; M0's gate not passed; the shard notice's description of `probe_val.bin` corrected |
| 2026-10-19 | `2026-10-19.md` | X17 matched rerun: the +0.0444 was an eval-mask artifact and the benefit is undetectable; entity decoding's 3× is circular, and the first holdout run is canceled at 11/20; run registry and publisher fixes; written 2026-10-21 from the commits |

