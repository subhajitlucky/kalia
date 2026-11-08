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
| D8 | 2026-09-22 | Adopt QK-Norm + logit soft-cap | Soft-cap only; QK-Norm only | X31 (2026-10-21) at two fresh seeds: QK-Norm alone **−0.0844**, soft-cap alone **+0.0099** and not additive | QK-Norm validated; soft-cap not supported (D55) |
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
| D20 | 2026-09-23 | Adopt "punch above weight" strategy: token efficiency first (code mix), sub-billion architecture wins (depth, GQA, block sharing), data quality, recipe; queue X1–X5 micro-ablations | Scale params; ignore literature | Modern sub-1B models train at 10k–30k tokens/param; our 43 tok/param shows the real gap. See docs/research/2026-09-23-punching-above-weight.md | active |
| D21 | 2026-09-23 | **Distillation rejected.** KALIA stays a pure from-scratch lineage; capability ceiling accepted knowingly | Silent distillation; reject outright | Ownership is the project's identity: every weight is ours and the recipe is auditable. Borrowed capability would compromise the core claim. Revisit only as a separately-labelled KALIA-D if ever wanted | closed: no distillation |
| D22 | 2026-09-23 | Adopt the high-impact techniques research track: test-time training (in-place fast weights), self-teaching (EMA teacher), retrieval at deployment. Refuse benchmark leakage and unlabelled claims | Pursue unverified 100× claims; ignore test-time compute | Measured multipliers are 1.5–8× (data-side mostly); TTT/Titans is the only technique that gives small models orders-of-magnitude on memory-bound tasks | active |
| D23 | 2026-09-23 | **Invention target set: KALIA-EM+TTT** — entity memory + in-place test-time fast weights at 58M for story consistency, evaluated on frozen probe stories | Pure architecture search without a target | Combination unpublished at our scale; cheap (~20-line in-place TTT); measurable and pure | queued (X7–X9) |
| D24 | 2026-09-23 | **Looped/recurrent depth promoted to front of queue (X11)**: 5 layers × 2 loops vs 10 unique layers; effective depth decoupled from parameters | Continue current dense 10-layer design | Ouro 1.4B matches up to 12B standard models on composition/reasoning; MobileLLM depth>width; our 10 layers is shallow for 58M | active |
| D25 | 2026-09-23 | **Reject RL self-play RSI and DSA/MoE at our scale, on evidence.** R-Zero collapse documented at 0.6B (peaks iteration 1, pseudo-labels 79%→63%); DSA useless at 1K context; MoE needs scale | Try RSI anyway; adopt DSA/MoE for prestige | Never adopt unmeasurable complexity; record negatives. See docs/research/2026-09-23-adoptable-frontier.md | closed |
| D26 | 2026-09-23 | Adopt continual-learning composition recipe (replay + self-distillation + importance reg + merged LoRA) for the v1 post-training stage | Naive sequential SFT | 28× retention improvement (1.2% → 34.9%) in a 100-task factorial study | planned |
| D27 | 2026-09-23 | **Publishing policy**: minimum-footprint, honest-only. Code source filtered to permissive licenses ("CodeParrot" had no license metadata). No data redistribution; attribution for all sources; weights Apache-2.0; publish at milestone (v0.1.2 complete), never chat transcripts | Publish everything now; republish datasets | User requirement: no legal exposure, no fake claims. Checklist at docs/public/publish-checklist.md | active |
| D28 | 2026-09-23 | Rebuild the v2 code shard with the permissive-license filter and remix (`kalia-prep-v2b`); all training intended for public release must use compliance-clean data | Ship v2 as-is (unfiltered code shard) | The first v2 build predates the license filter; the public model cannot contain unclear-license code. Remix reuses existing shards, ~20 min CPU | in progress |
| D29 | 2026-09-23 | Add an evaluation harness: frozen 24-prompt probe set, held-out sentence loss/bpB, side-by-side model comparison tool, and a public reproduction notebook | Rely on val loss and ad-hoc samples | Pre-publication rigor: recipe changes get a deterministic quality check plus qualitative comparison; others can verify the ablation claim on free compute | done |
| D30 | 2026-09-23 | **Reasoning-model policy**: adopt published methods (GRPO/RLVR/long-CoT SFT) with our own code and weights; never start from external weights. R1-style RLVR deferred at 58M on evidence (RLVR degraded accuracy at 135M; self-play collapsed at 0.6B; no verifiable reward for stories) | Copy R1 weights; attempt RLVR now | Methods are public science; weights are the ownership line. See docs/research/2026-09-23-reasoning-models-policy.md | active |
| D31 | 2026-09-23 | Freeze Muon+ LR at 0.02 for the 58M recipe; LR sweep complete | Continue searching LR | Sweep result: 0.015/0.02 tie at ~3.494, 0.03/0.06 degrade; further LR tuning has no headroom | closed |
| D32 | 2026-09-23 | Adopt three ancient-text-inspired algorithm designs (A1 Kautilya adaptive mixture, A2 Utsarga-Apavada schedule, A3 Apoha-ELECTRA) as testable experiments X13–X15, Apoha-ELECTRA first | Dismiss the idea outright; adopt without controls | Each design states its modern relative and the novel combination; all are micro-ablation testable. See docs/research/2026-09-23-ancient-texts-algorithms.md | queued |
| D33 | 2026-09-23 | Adopt four more designs from Vedic recitation, Yoga, Natyashastra and Jain logic: G1 Jata-patha reverse-chaining loss (X16), G2 Samyama inference-time loop scaling (X17, eval-only), G3 Rasa-conditioned generation (X18), G4 Anekantavada two-seed soup (X19) | Metaphor-only use of the texts; discard | Vedic patha methods are literal redundancy/chaining engineering; G1–G3 yield testable mechanisms with measurable predictions. See docs/research/2026-09-23-god-texts-algorithms.md | queued |
| D34 | 2026-09-23 | Adopt three Veda-derived designs: H1 Nasadiya self-knowledge head (X20), H2 Samaveda format-conditioned generation (X21), H3 Yajurveda procedural subset (X22); H4 Rigveda ordering recorded as backlog | Treat the Vedas as metaphor only | Each Samhita has a distinct function that compiles to a mechanism; H1 has a clear modern relative (selective prediction) and a new twist (token-level self-knowledge during pretraining). See docs/research/2026-09-23-vedas-algorithms.md | queued |
| D35 | 2026-09-23 | Gita compilations: adopt X23 Guna mode controller (adapted, **cited prior art** Antahkarana-Net) and X24 Samatva aux-loss balancing; note Nishkama stopping and Svadharma adapters as already implemented/planned; add a process-impact statement to the model card | Claim the guna controller as novel; discard the Gita material | Prior art exists and is validated at small scale — adaptation, not invention, and must be labelled as such. Honesty preserved. See docs/research/2026-09-23-gita-algorithms.md | queued |
| D36 | 2026-09-23 | Mahabharata audit: build entity-consistency evaluation harness (`eval_entities.py`); **narrow the KALIA-EM+TTT claim** — entity memory has direct prior art (Papalampidi & Lapata 2022), so the residual novelty is the EM+TTT combination at 58M, from scratch, on free compute; nested-narration and sentiment-arc ideas recorded as backlog | Keep the original broad EM novelty claim | Computational network analyses of the Mahabharata and entity-memory literature exist; citing them protects the project's credibility. See docs/research/2026-09-23-mahabharata-algorithms.md | active |
| D37 | 2026-09-23 | **Chakravyuha dismissal corrected.** Build the Abhimanyu-gap metric (`eval_reversibility.py`); redesign X16 as chunk-preserving reversal training (entity-preserving / random-segment), which also doubles effective tokens in our data-bound regime; adopt the project rule "know the exit before you enter" (exit/rollback criteria defined before any experiment starts) | Keep the dismissal; keep X16 as a reverse-prediction head | Reversal-curse literature shows data-level chunk-preserving reversal is stronger and validated; the metric operationalizes entry-exit asymmetry. See docs/research/2026-09-23-chakravyuha-mechanism.md | active |
| D38 | 2026-09-23 | Jagannatha audit: adopt **hash-anchored pre-registration** (J1: mustard-seed trail) starting with X16, and **pre-registered stopping as identity** (J2: unfinished by design) for all remaining experiments. No novel-algorithm claim; prior art cited (`preregister` tooling; Hofman et al. 2023) | Continue ad-hoc experimentation; claim pre-registration as original | The two practices are what make any future "invented algorithm" post defensible; the texts supply the discipline and the narrative. See docs/research/2026-09-23-jagannatha-algorithms.md | active |
| D39 | 2026-09-23 | **Adopt the strategic audit** (`docs/plans/2026-09-23-strategic-audit.md`): finish v0.1.2 with one final session (target 2.45), run pre-registered X16 (concurrently if allowed, else after), run the zero-GPU evaluation suite, publish the milestone, then start v0.2.0 only after quota reset with a frozen recipe. **Freeze the experiment queue** (no new families before v0.2.0) and pause ancient-text audits until the first post is live | Stop v0.1.2 now and bank ~7 GPU-h for v0.2.0; run more queued experiments (X1–X26) first; keep expanding the research queue | Quota ≈16 GPU-h remaining; the cosine-decay tail is where final quality arrives, so a designed finish produces the strongest publishable artifact; the queue is 26 experiments ahead of execution — execution focus is the binding constraint now | active |
| D40 | 2026-09-23 | **Quota-exhaustion response**: X16 deferred to the Saturday reset (I12); v0.1.2 stops today at ~step 3,485 (~73% of schedule, checkpoint auto-pushed); session 4 (~6.2h) plus X16 (~1.7h, concurrent if allowed) run Saturday after reset; publish assets are finalized today (zero GPU) and the milestone publishes after session 4 completes; v0.2.0 follows (~3 sessions) | Publish v0.1.2 at 73% today; abandon the remaining cosine; run X16 before finishing v0.1.2; wait for reset doing nothing | LR at 73% is still 26% of peak — the remaining cosine tail holds the largest expected gains (the step-2750 drop already showed the decay working), so finishing is worth 6.2h; publish-asset work costs no quota | superseded by D41 |
| D41 | 2026-09-23 | **v0.1.2 declared final at step 3,478** (quota-stopped at 73% of schedule; plateau documented): the last ~700 steps produced no measurable gain (deterministic probe +0.055 regression, val flat within ±0.1 noise for 1,000 steps), so the remaining 6.2h buys nothing measurable and is better spent on v0.2.0; Saturday = X16 (1.6h) + v0.2.0 start. High-precision val (CPU, free) runs as a safety valve; if it contradicts the convergence reading, this decision is revisited before Saturday | Finish session 4 anyway to complete the cosine; publish later | Evidence over ritual: the cosine is a means to gains, and the gains are measurably absent; stopping is reversible, this week's quota is not. First application of the J2 pre-registered-stopping principle. Also: **v0.1.2 uses plain Muon** — `muon_plus` was never in the full-scale config (docs corrected; v0.2.0 pre-registration amended via hash-registered Amendment 1) | active |
| D42 | 2026-09-27 | **X16 rejected; reversal transform not promoted to v0.2.0.** Seed-pair result: gap 5.2465 vs control 5.1107 (Δ +0.136, wrong direction) and forward val 3.9641 vs 3.8816 (Δ +0.083) — both pre-registered bars missed on both metrics across both seeds. v0.2.0 launches on the frozen recipe + compliance-clean v2b data. P1 (gap > 0.3 nats) confirmed at micro scale (~5.1 nats, consistent with 6.06 at 58M). Token-level reversal or lower ratios recorded as a separate future hypothesis | Promote anyway (the transform "should" help); silently drop the experiment; retry with different hyperparameters and report only the win | The hashed promotion rule requires P2 ∧ P3; neither held. The negative result is published with the same prominence as a win would have been — that is the entire point of pre-registration. See docs/journal/2026-09-27.md | closed: negative result |
| D43 | 2026-09-28 | **`kalia-prep-v2b`'s val.bin is the canonical held-out set; v0.1.2 re-baselined on it at 3.0533 (0.9992 bpB, 819,200 tokens — the same 100-batch protocol as the originally published figure).** Found when v0.2.0's first val point (4.8272 @ step 250) appeared 0.74 nats worse than v0.1.2's (4.0881) while its train loss was better. The published v0.1.2 weights score 2.3915 on the old val set and 3.1070 on the new one, so the val set alone explains the gap and v0.2.0 is not behind. The old figure 2.4366 / 0.8184 bpB stays true *for the old val set* and must not be tabulated beside a v2b number | Treat 4.8272 as a real regression and debug the model; re-train v0.1.2 to get a matching number; quietly restate 2.4366 as if the sets were interchangeable | A corpus rebuild silently redefines the yardstick, so any number crossing a rebuild boundary is re-measured on both sides before it is believed — ~11 minutes of CPU here, versus a false regression narrative that would have redirected the whole cycle. Within-version comparisons (the J2 stopping rule) are unaffected | active |
| D44 | 2026-09-28 | **Licence finding (I16) and the data-publication policy it forces.** A 20k-file sample of the code corpus measures 41.7% of characters under non-permissive licences, 39.5% of all files GPL-family. v0.1.2's code slice was built without a licence filter (prep-v2 predates the filter by 40 min), so ~50M copyleft tokens, ~2% of the training set, entered that checkpoint. Two bindings consequences. **(1) v0.1.2's public card must disclose it** — the card had listed three of four sources and omitted the code slice entirely, i.e. the one dataset that needed disclosure. **(2) 'publish all training data' is redefined as 'publish the filtered corpus plus the filter', never the raw corpus** — redistributing copyleft text is what triggers the obligation, and v2b is licence-clean by construction. Weights-only releases are unaffected on the prevailing view that training does not make weights derivative | Leave the card as-is because the release does not redistribute data; unpublish v0.1.2; retrain v0.1.2 on filtered data | The interpretation was committed in `a9ddb81` *before* the number was known: material share means the card leads with the rebuild. It is material by a wide margin, and the filter already exists and is tested — the defect is purely that it was created 40 minutes after the corpus that needed it, with nothing tying a new filter to a rebuild of existing datasets. Disclosure is owed regardless of whether an obligation is technically triggered. | active |
| D45 | 2026-09-28 | **Accuracy thresholds must be stated against the metric's measured noise floor, and our existing accuracy thresholds are under-powered.** X18's CPU completion closed F-3 on its registered terms (gated +1.20 pp on WinoGrande against a >=0.5 pp bar) and the result is still worthless: lm-eval's standard errors at `limit=500` are 1.99-2.24 pp, so the bar sits ~4.5x below the noise and the observed delta is 0.54 sigma. A 0.5 pp bar at that sample size passes about half the time by construction. This is recorded against our own pre-registration, not the mechanism. **The registered thresholds are not retroactively changed**: S-A is still judged on its 1.0 pp term exactly as written, and the noise floor is published alongside the verdict rather than used to rescue it, the same handling P-C was given. Interim v0.2.0 regressions of ARC-Easy -4.6 and LAMBADA -4.6 are outside noise and stand; PIQA -0.2 and WinoGrande -0.8 are inside it and do not. Every future accuracy bar is set above the measured standard error or uses enough samples to bring it below | Move the bar now that X18 failed it; silently re-run the benchmark with more samples and pick the number that rescues S-A; keep 0.5 pp because it is a round number | A threshold chosen without reference to the metric's resolution is not a threshold, it is a coin flip with extra steps. X18's loss bar (0.010 nats) was sound -- the failure is specifically that we wrote an accuracy bar twice without ever measuring what one standard error is worth at our sample size. The honest cost is that X18's accuracy half produced no information and had to be re-planned as X19's loss-only question | active |
| D46 | 2026-09-28 | **The gated-residual gain is real but is not a gate.** X19 ran one arm: the four-branch residual normalisation with the gate deleted, 500 steps, same seed/corpus. Result 4.7680 against control 4.7662 and gated 4.7226, against a pre-registered G-1 bar of 4.7544. **G-1 fails by 0.0136: the branch normalisation contributes nothing and every bit of the -0.0436 lives in the gate.** G-0 had measured the gate's response to its input at 5e-06; G-1 shows removing it destroys the gain. Both are correct, and together they mean the module is a learned *static* per-channel modulation -- effectively a rank-32 low-rank adapter on the residual path -- rather than the data-dependent read it was borrowed as. The component earns its parameters; the name and the claim do not transfer | Ship Gated Residual as a validated mechanism; call the result a null and drop the arm; re-run with a second seed before believing any of it | A technique can be necessary and inert with respect to its documented mechanism simultaneously, and only a controlled removal says which half you are measuring. Honest report: the name transferred, the behaviour did not, and the next experiment this creates is a static low-rank residual adapter with no data dependence at all -- if that reproduces -0.0436, the gate was never a gate | active |
| D47 | 2026-09-29 | **X20: a static learned per-channel modulation beats the published data-dependent gate by 0.0922 nats with 73,728 fewer parameters. Nothing is promoted; replication is pre-registered.** The arm keeps the branch normalisation and replaces the gate's input path with a learned constant. Result 4.6304 against control 4.7662, gated 4.7226 and branchnorm 4.7680, clearing H-1's bar of 4.7544 by 0.1240. The input-dependent read is therefore not inert at our scale but actively harmful, and the component that helps is not the mechanism the source technique describes. H-3 registered 'lands strictly between the known arms' and failed below both, which is recorded rather than argued away. **But 0.1358 is 3.1x the largest effect any architecture arm has produced here, every arm so far is single-seed, and the previous best was already shown unattributable -- so the number is reported and nothing ships.** A paired replication against control at fresh seeds is pre-registered as X21, on the discipline the modded-nanogpt records use for a result of this size: interleaved unseeded repeats of arm and control, with a stated test | Ship it to v0.3.0 on one seed because the effect is large and the direction is unambiguous; discount a large effect as noise because one seed is too few; drop the whole line as a single-seed mirage without replicating | A big effect from a component simpler than the one it beats deserves more scepticism, not less -- we rejected 0.0436 on exactly these grounds and 0.1358 on one seed is less established, not more. H-2 also remains unverified: the kernel never emitted the static gate's channel dispersion, so whether the adapter learned anything non-trivial is unknown, and that is part of what the replication must close | active |

> **This index covers D1–D47 only.** D48 onward have full entries below but no
> row here, so a reader scanning the table for the newest decision finds the
> wrong last entry. Noted 2026-10-18 during a whole-record audit and left as a
> known gap rather than back-filled: each row is a paragraph, and reconstructing
> six of them from their bodies is the kind of summarising that drifts from the
> thing it summarises. **Read D48–D53 in the body, not the table.**

## D48 — The architecture line is closed; there is no baseline, so deltas were unreadable (2026-09-29)

**Context.** X21 replicated X20's static-gate arm at two fresh seeds. The
−0.1358-nat gain inverted to **+0.0516** (worse than control) at both seeds, and
R-1/R-2/R-4 all failed. Running it also produced the number this project never had:
the control's own spread across configurations we had been treating as
comparable.

**Evidence.** Seed 1337 gives control 4.7662 (X18) and 4.7680 (X19) — two
independent sessions, **0.0018 apart**. Seeds 1338/1339 give 4.8769/4.8908, about
**0.11 higher**. Session noise is therefore tiny and the ~0.11 gap is a seed
effect. Within the fresh pair the two seeds differ by only 0.0139, so seed effects
are not exchangeable: some seeds are simply much better.

**Decision.**
1. **Do not ship the static gate, or any of the gated-residual variants.**
2. **Close the architecture-ablation line.** Not because the idea was wrong —
   because the experiment was uninterpretable without a variance baseline, and
   the baseline turns out to be larger than the effects being chased.
3. **No further architecture arm may be run from a single seed.** Minimum design
   going forward: paired control and treatment at ≥2 fresh seeds, with the control
   spread reported as a first-class result before any delta is interpreted.
4. **Retrospective correction to earlier claims.** X18's gated residual
   (−0.0436) and X19's branch norm (−0.0018) are both **smaller than baseline
   noise** and are now classified as null results, not as findings ranked by size.
   The 0.01917 gate mean and its 5e-06 input dispersion remain valid
   measurements; what they were read as supporting was not.

**Consequence for v0.3.0.** The static gate is removed from the v0.3.0 plan. The
budget that was going to it moves to the continual-update corpus and the replay
sweep, which have never been measured on a shared baseline and so have not yet
produced a number that could have been noise.

**Cost.** ~1.2 GPU-hours of the 1.76 remaining, spent to delete a component.
Cheaper than shipping one, and cheaper than the alternative: X20's 0.1358 was
going into the model card as the project's best result.

## D49 — A load-bearing citation was about the wrong paper; the decision survives on weaker grounds (2026-09-29)

**Context.** Before building v0.3.0's continual update, every arXiv ID in
`docs/research/` was checked against arxiv.org. The continual-learning design
document cited arXiv 2403.08763 under the title *"Rules of thumb for continual
pre-training"* — **a title that does not exist**; the real paper is *"Simple and
Scalable Strategies to Continually Pre-train Large Language Models"* (Ibrahim et
al.). More seriously, it attributed to MIITA (arXiv 2607.22556) the finding that
replay degrades sharply as the backbone shrinks, including a quoted phrase, a
CT0/FOREVER degradation claim, and a 0.6B floor. **MIITA's abstract contains none
of that** — it is a memory-based, inference-time method that works "without
backbone updates," not a replay paper at all. The 0.6B figure appears to have
been inferred from the words "small language models" in its title.

**Decision.**
1. **Retract the MIITA-derived claim and the "slope goes against us" conclusion.**
   Both are recorded as retracted in the research doc, in place, with the
   retraction visible rather than a quiet edit.
2. **Keep the replay-ratio sweep — for a weaker, honest reason.** The recipe is
   verified at 405M and 10B. Nothing verified covers 58M. Therefore no ratio can
   be *adopted*; only measured. The decision was always right; the reason was
   partly wrong, and the surviving reason is weaker but sufficient.
3. **Do not build v0.3.0 on any unverified claim.** The four CL papers are now
   abstract-verified. The active-forgetting and orthogonal-projection methods are
   marked direction-setting, not evidence.
4. **`test_citations.py` pins the verified titles** and fails if a research doc
   cites an ID with no verified title. Falsified three ways: re-inventing the
   Ibrahim title, substituting a wrong FST title, and restoring the MIITA claim —
   all three fail the suite, the clean file passes.

**Cost of getting this wrong.** If v0.3.0 had been built and shipped on the
MIITA citation, the model card would have claimed a literature-grounded rationale
that did not exist. This is the same failure as X20 in a different costume: a
confident number resting on something that was never checked. D48 and D49 have
the same shape, and that is now the project's characteristic risk — an unverified
input that looks like evidence. The mitigation is the same in both cases:
hash-register the claim, and require replication before it ships.

## D50 — The replay default's justification was unlocatable; removed rather than restated (2026-09-29)

**Context.** While building the Kautilya mixture loader, I checked the citations
in code comments. `data.py`'s `MixtureDataset` justified our 10% replay default
with: *"Bethune et al. (ICML 2025) measured that injecting as little as 1% of the
pretraining data into the finetuning mixture is enough to prevent forgetting."*
**That citation could not be located on arXiv or Semantic Scholar on 2026-09-29.**

**Decision.** Remove it and say so in place, rather than searching for a similar
real paper and attaching that one's numbers to the claim — which is precisely how
D49 happened. The default itself is unchanged and now rests on what is verified:
Ibrahim et al. (arXiv 2403.08763) establish that replay is *part* of a recipe
sufficient to match full retraining, at 405M and 10B, without a validated ratio in
the abstract. 10% is the unremarkable midpoint of the registered 0/10/40 sweep,
not a number anyone has established at 58M.

**Why now and not later.** One fabricated title and one misattributed finding
have already shipped this month. This is the third instance of the same shape and
it was sitting in a code comment, where nobody re-reads it.

**Consequence.** `test_citations.py` asserts the unlocatable citation appears
only inside its retraction note, and that live text justifies the default with the
verified paper. The first version of that test passed with the fake citation
reinstated as live justification, because it checked "phantom absent OR note
present" and the note was elsewhere in the file — so it is written to isolate the
note and compare against what remains. Verified by reinstating the fake (fails)
and by deleting the note (fails).

## D51 — `--resume` silently continued the source schedule; the re-warm was never on (2026-09-29)

**Context.** Building the v0.3.0 execution notebooks surfaced a defect that would
have voided Step 2 of the pre-registration without producing a single error.

`train.py --resume` continues the source run's cosine schedule. v0.2.0 ends at
step 4770 of `max_steps` 4770, and `base_lr_scale` returns `min_lr_ratio` for any
step at or beyond `max_steps`. So a resumed update would sit at **LR multiplier
0.100 for every step** — the re-warm arm would have been byte-identical to its
no-re-warm control, and Step 2 would have reported "the recipe does not transfer
at 58M." **A null for a mechanism that was never switched on.** Measured: 0.100
flat, versus 0.02 → 0.97 → 0.10 once the schedule is rebased.

**Two intermediate fixes were worse than the original, and both are recorded
because both would have produced a plausible wrong answer:**

1. Rebasing the schedule unconditionally also re-warmed the **control**. Both arms
   peaked at 0.00060 — two identical arms, meaningless null.
2. Gating the step budget on `rewarm` gave the control a 40-step budget against a
   `start_step` of 60, so it trained **zero** steps while the treatment trained 40.
3. Making the budget origin-relative for any resumed run broke the smoke test
   asserting `--max-steps` is an absolute ceiling, and would have silently
   changed how every already-recorded resume behaves.

**Decision.**
1. `--max-steps N` keeps its historical meaning: absolute step N. This is asserted
   by an existing smoke test and must not move.
2. A new config key **`update_budget`** means "N more steps", and is set on all
   seven v0.3.0 arms. Without it a resumed v0.2.0 run with `max_steps` 500 trains
   nothing, because `step < hard_max_steps` is false on entry.
3. **`rewarm` now defaults to `False`.** A plain `--resume` must not change the
   schedule of runs whose results are already published. The opt-in is what keeps
   v0.2.0's history reproducible.
4. The schedule origin is rebased **only** when `rewarm` is set, so the control
   arm continues the source cosine and the treatment restarts it.

**Verification.** Both arms run end-to-end on CPU: same 40 update steps, re-warm
peaking at 0.00060 against the control's 0.00006 — a 10× difference in the
learning rate each arm trains at. 12 tests in `test_rewarm_schedule.py`, four of
which perform real resumes. Falsified by restoring the unconditional rebase: two
fail.

**Standing lesson.** This is the third time an unexercised code path has turned
out to be wrong (the pyflakes gate, `per_source_loss`'s model signature, and now
the resume schedule). In every case the defect was invisible to reading and to
unit tests, and visible only to running the thing. The v0.3.0 arms are the first
part of this project whose execution path is exercised end-to-end before it is
trusted with quota.

## D52 — The v0.2.0 card denied a redistribution the repository actually ships (2026-10-17)

**Context.** An external audit of the published artifacts found that
`kalia-lm/kalia-v020` contains `corpus/{fineweb,tinystories,cosmopedia,python}.bin`
(4 × 249 MB, 124,500,691 tokens each), `corpus/probe_val.bin` and
`corpus/manifest.json` — the equalised per-source shards and probe set published
on 2026-10-07 for the continual-update experiments (journal 2026-10-08). The
model card published the same day still said "Datasets are not redistributed
here", as did the repo README and the publish checklist. The upload was
deliberate; the disclosure was missing. This is the fourth instance of the D49
shape: a confident public claim that a later change quietly made false.

**Evidence.** All five published shards' LFS SHA-256 match
`corpus/manifest.json` and `docs/legal/val-shard-hashes.json` exactly. The
licence matrix had already concluded that redistributing the v0.2.0 corpus is
"yes, conditional" (CDLA-Sharing notice, ODC-By attribution, Apache notice,
filtered code) — the card contradicted the matrix, not the licence. The audit
also found the card's YAML `datasets:` listed `sedthh/gutenberg_english` (an
eval-probe source, not training data) and omitted Cosmopedia.

**Decision.**
1. The v0.2.0 card now discloses the released shards with per-file token counts,
   sources, licences and attribution, and states that the full 2.4B-token
   mixture remains undistributed.
2. The repo README and the publish checklist are amended to the conditional
   policy: raw data is never republished; only hash-pinned, attribution-complete
   shards may be, and the checklist names the v0.2.0 `corpus/` release as the
   sanctioned example.
3. `tools/publish_model_card.py` now takes `--release v012|v020` (default v012,
   so historical behaviour is unchanged). The v0.2.0 card was re-published and
   verified byte-for-byte by round-trip download.

**Standing lesson.** A published repository is a claim about its contents. When
a later step adds files, the card that describes the repository has to move in
the same change — the corpus upload and the card edit were minutes apart and
still drifted. The audit that found this also confirmed the artefact side is
sound: shards hash-match their pins, all Kaggle kernels and datasets are private,
no copyleft bytes are published, and the only credential at rest is the
`kalia-publish-aux` Kaggle dataset, which holds the publish token by design and
must never be made public.

## D53 — WSD lands as mechanism; the decay shape stays a knob, not a finding (2026-10-18)

**Context.** E11 in the 2026-09-23 technique backlog — "WSD schedule +
batch-size warmup", sourced to Kimi K2, GLM-4.5 and IMU-1, expected gain
medium, risk low — was recommended and never landed. The bill came due:
v0.2.0 finished its full 4,770-step cosine and **plateaued for its last 31%**,
so roughly 6.3 GPU-hours bought no measurable val movement. A cosine spends its
whole run decaying; WSD holds the peak and pays for the anneal once.

**Evidence.** The plateau is in the record (journal 2026-09-28) and in v0.2.0's
own val curve. The mechanism is a schedule shape, so it is verifiable without a
GPU: the tests pin the cosine to a frozen reference implementation at eleven
steps, assert WSD holds exactly 1.0 across the stable phase, assert it is
monotone after warmup, assert it lands on `min_lr_ratio` at `max_steps`, and
assert `wsd_lr_scale(step) >= cosine(step)` at every step of a 1,000-step run —
with a margin > 0.3 in the region where the cosine has already given up. That
last one is the actual claim: WSD cannot spend less of the budget than the
cosine it replaces.

**Decision.**
1. `lr_schedule: wsd` is implemented in `train.py`, opt-in, and **enabled on no
   registered arm**. Cosine remains the default and is bit-identical when the
   key is absent — every recorded run in this project trained under it, and the
   regression test says so rather than assuming it.
2. The stable phase runs to `max_steps - wsd_decay_frac x max_steps`
   (default 0.1), then anneals to `min_lr_ratio` once.
3. `decay_shape` in {linear, cosine, inv_sqrt} exists because **the decay shape
   is unmeasured at 58M here**, and the default is `linear`: the assumption-free
   choice. The convex decay the WSD literature favours is available as
   `inv_sqrt` and is deliberately not the default, because inheriting it would
   be the same unmeasured claim D48 closed the architecture line over.
4. The target-loss reactive decay composes with WSD unchanged, and that
   composition — a run that hits its target at step 300 paying 100 steps of
   anneal instead of the 700 the cosine had already scheduled — is the
   "flexible decay timing" E11 actually asked for. It is tested on the WSD path
   rather than inferred from the cosine path.
5. `validate_schedule` refuses an unknown schedule, an unknown shape, a decay
   fraction outside (0, 1], and a WSD config with no stable phase — before any
   GPU work, not at the first learning-rate lookup.

**Stated limit.** The three sources recorded for E11 are model reports. The
primary literature on *decay shape* was not consulted for this landing — the
research search endpoint returned an authentication failure, which is a tooling
matter and is not worked around by guessing. So the mechanism is implemented on
the backlog's authority and the shape is explicitly left for measurement; no
claim is made that `linear` beats `inv_sqrt` at 58M. The first WSD run should
treat the shape as an arm, not a constant.

## D54 — The record is the product, so the record is now checked (2026-10-18)

**Context.** A whole-record audit — six parallel reads of the journal, decisions,
pre-registrations, results, research backlog and public documents, 72,000 lines —
found that the project's most distinctive asset had drifted from its own
evidence. The most-churned files in 214 commits are `DECISIONS.md` (36),
`README.md` (25), the journal (25) and `LEDGER.md` (20), against `train.py` (18)
and `model.py` (13). The apparatus that makes the work worth reading was the
least-verified thing in it.

**Evidence.** Nine public-claim defects, none in the same place as another: the
v0.2.0 card called the released `corpus/probe_val.bin` "held-out windows" while
the shard notice — corrected the same day — said otherwise; the root README
called v0.2.0 "not yet public" while linking to it, and had X18's attribution
*inverted* against D46; both cards omitted the code corpus from their HF
`datasets:` and named it `bigcode/codeparrot-clean` where `prepare.py` loads
`codeparrot/codeparrot-clean` — the mismatched name also sat in the licence
audit's own provenance record; the licence matrix denied a redistribution two
rows above its own entry recording it; and the accuracy framing claimed "three of
five movements inside the noise" where only one is inside one standard error and
none reaches two (D45's measured 1.99–2.24 pp).

**Decision.**
1. The nine are corrected, with the original wording left visible beside its
   correction wherever the document is a record rather than a claim.
2. `tools/audit_claims.py` checks the four invariants that would have caught
   them, and CI runs it: dataset attribution complete against the shard notice,
   `probe_val.bin` never described as held-out, every cited `D<n>` defined, and
   the released val/bpB matching the artifact that was re-scored. Each rule names
   the defect it prevents; a rule that cannot is not added.
3. `tools/check_notebooks.py` is wired into CI. It existed, passed, and was
   invoked by nothing but its own docstring — the anti-decorative-gate tool was
   itself decorative, which is the eighth instance of a class the project keeps
   re-finding.
4. **Frozen documents are not corrected in place.** The same false accuracy
   sentence stands in `2026-09-28-v020-final-evaluation.md`, which is
   hash-anchored and read by the ledger as `extended: +78/-0`. Editing it would
   flip that classification to `EDITED`, and spending the one signal that says
   "registered text changed" on a wording fix is a bad trade. SYNTHESIS records
   where the false sentence still lives instead.

**Standing lesson.** Every root cause in this project's 24 incidents that has a
*named tool* is closed; every one whose fix is a *convention* is open. This entry
moves four conventions into tools. The three that remain open — pin the code
bundle rather than fingerprint it, state a threshold above a measured error bar,
never train on the measuring set — are named in the journal's Open list, and each
is waiting for the same treatment.


## D55 — QK-Norm stays; the logit soft-cap does not (2026-10-21)

**Context.** D8 adopted QK-Norm + logit soft-cap on one 30M run at seed 1337
(Muon 3.5937 → 3.5103, a further −0.0834). It was never replicated and the two
components were never isolated. The 2026-10-18 audit found the number below the
cross-configuration 30M control spread (0.1139) and above the paired control
spread (0.0139), and the cards still cited it.

**Evidence.** X31 ran the 2×2 at seeds 1338/1339, 763 steps, one kernel:

| contrast | seed 1338 | seed 1339 | mean |
|---|---|---|---|
| qk − control | −0.0921 | −0.0766 | **−0.0844** |
| cap − control | +0.0129 | +0.0069 | **+0.0099** |
| both − control | −0.0839 | −0.0743 | −0.0791 |

Q-1, Q-1b, Q-2a and Q-3 pass; **Q-2b fails**: the soft-cap alone is slightly
worse than control at both seeds. The bundle is QK-Norm, not an interaction.

**Decision.**
1. **QK-Norm is validated at two seeds** for the 30M micro-recipe; the shipped
   claim is restated as QK-Norm's −0.084, not the bundle's −0.08.
2. **The logit soft-cap is not supported.** Future configs keep `qk_norm` and
   drop `logit_softcap`. The shipped v0.2.0 is not retro-changed; it was trained
   with both, and the weights are what they are.
3. The cards, SYNTHESIS and the X31 pre-registration carry the attribution.
4. This is a 30M micro result. Whether the 58M model benefits from the soft-cap
   is untested; with no positive evidence and a measured +0.010 cost, the
   simpler recipe is the default for the next run.

**Cost.** 3.8 GPU-h for eight arms, inside the 2026-10-21 quota window. The
verdict is computed from the archived curves by `tools/x31_verdict.py`, not
transcribed.

**Status.** active.

## D56 — The ARC-Easy regression was a 500-sample artifact (2026-10-21)

**Context.** The v0.2.0 release verdict was measured at `limit=500`, where D45
measured lm-eval's standard error at 1.99–2.24 pp. The 2026-10-17 review
recommended 2,000 samples; `eval_bench.py` changed its default on 2026-10-18, but
no 2,000-sample run was archived. The public cards still said the corpus change
"cost 3.8 points on ARC-Easy" and that two benchmarks regressed.

**Evidence.** `kalia-bench-pair-2k` scored both checkpoints in one CPU session at
2,000 samples (PIQA 1,838 and WinoGrande 1,267 use their full sets):

| task | v0.1.2 | v0.2.0 | Δ (pp) | z |
|---|---|---|---|---|
| PIQA | 59.63 | 61.37 | +1.74 | 1.08 |
| HellaSwag | 35.60 | 35.60 | 0.00 | 0.00 |
| WinoGrande | 50.28 | 50.75 | +0.47 | 0.24 |
| LAMBADA | 25.15 | 23.90 | −1.25 | −0.92 |
| ARC-Easy | 45.95 | 46.40 | **+0.45** | 0.29 |

The 500-sample ARC-Easy −3.80 did not replicate; the 500-sample HellaSwag +3.00
became 0.00. No movement reaches 2σ. LAMBADA is the only task outside the 1.0 pp
allowance, by 0.25 pp.

**Decision.**
1. The **registered 500-sample verdict is not edited**: S-A FAIL on ARC-Easy and
   LAMBADA remains what the frozen pre-registration produced at that sample size.
2. Public cards, README and SYNTHESIS now report the **2,000-sample** comparison:
   S-A fails on **LAMBADA only**; the ARC-Easy regression is retracted. The 500
   numbers remain in the record with their ~2 pp error.
3. v0.2.0 remains an **experiment release, not a promotion**: the registered
   rule still fails at 2,000, even though the failure is one task at the edge of
   the allowance and not statistically resolvable.
4. The loss/accuracy lesson is restated: a 0.2285-nat loss gain produced no 2σ
   accuracy movement and LAMBADA −1.25; it did **not** cost 3.8 points on
   ARC-Easy. The mixture remains the open lever, but it is not the accuracy
   catastrophe the 500-sample run suggested.

**Cost.** ~3 h of CPU quota, no GPU. Artifacts:
`docs/results/bench-pair-2k/`; narrative:
`docs/eval/2026-10-21-v020-bench-pair-2k.md`.

**Status.** active.

## D57 — X32: the entity heads are necessary, not the switch (2026-10-22)

**Context.** The entity holdout showed the candidate pool contains a better
continuation than greedy (oracle 0.0618 vs 0.0170 on the unseen half), while every
cheap selector fails. X32 tested the mechanistic version of that finding: is a
specific attention head carrying the wrong-name signal, and does ablating it
improve the continuation?

**Evidence.** Entity attention is concentrated in layer 9: heads 3, 2 and 7 have
**3.45×, 2.37× and 2.12×** uniform enrichment; no other head clears 2×. But
ablating those heads changes second-half retention by **−0.0117**
[−0.0263, +0.0000] against greedy — they are necessary, not harmful. The random
three-head control is −0.0052 [−0.0291, +0.0199], and the reference NLL rises by
**1.05%**, so the ablation did not destroy the model. H1, H3 and H4 pass; **H2
fails**.

**Decision.**
1. **No more head ablation.** The single-switch hypothesis is closed at this
   scale; repeating ablations until one improves retention would be the
   threshold-below-noise failure in a new costume.
2. **The selection failure is downstream at the readout or distributed.** The
   next test is a **linear probe on layer-9 residual states**: can a simple
   readout rank the candidates better than the model's own log-probabilities?
   If yes, the information is linearly decodable and the problem is the output
   path; if no, it is distributed or absent.
3. Activation steering is the second test, only if the probe finds a direction.
   A pass needs replication before any capability claim.

**Cost.** One GPU session, twenty stories. Artifacts: `docs/results/x32/`;
registration: `2026-10-22-X32-entity-selection-circuit.md`.

**Status.** active.

## D58 — X33: the midpoint state linearly knows whether the entity will survive (2026-10-22)

**Context.** X32 found entity attention concentrated in layer-9 heads 3, 2 and 7,
but ablating them reduced second-half retention. The heads are necessary, not the
hiding switch. X33 tested the readout: show the model only the prompt plus the
first 128 generated tokens, capture the layer-9 residual, and predict the
candidate's second-half entity retention.

**Evidence.** Leave-one-story-out ridge probe (α = 10) on 1,024 features
(mean over the first half + last-token state), 20 stories, 8 candidates:

| selector | mean second-half retention |
|---|---|
| greedy | 0.0170 |
| log-probability | 0.0103 |
| first-half entity pick | 0.0277 |
| **linear probe** | **0.0324** |
| oracle | 0.0618 |
| shuffled-label probe | 0.0205 |

Probe − greedy: **+0.0155**, 95% CI [+0.0005, +0.0321]. Probe − log-prob:
+0.0222 [+0.0041, +0.0443]. Probe − half: +0.0047 [−0.0106, +0.0217].
Shuffled-label probe − greedy: +0.0036. The probe closes about **35% of the
coverage gap** (0.0324 against greedy 0.0170, oracle 0.0618).

**Decision.**
1. **A linear midpoint signal exists.** The information about future entity
   persistence is present before the second half, and a simple readout finds it.
2. **Do not call it a selector yet.** It beats log-probability clearly, but the
   +0.0047 advantage over the first-half rule includes zero at n=20. Replicate on
   a second story set and a second generation seed before any capability claim.
3. If both replications hold, train the probe on more stories and measure the
   **selective-accuracy curve** with abstention. If they do not, close the linear
   selector track and keep the readout result.
4. No more head ablation. X32 closed the single-head hypothesis; X33 moves the
   question to the readout.

**Cost.** One GPU session, twenty stories. Artifacts: `docs/results/x33/`;
registration: `2026-10-22-X33-linear-readout.md`.

**Status.** closed — X34 did not replicate (D59).

**Update 2026-10-22.** X34 v1 was invalid: the stager silently left
`SEED_BASE = "7"`, so it reran X33's sampling seed. The stager now pins and
reads back `SEED_BASE` with a test. X34 v2 is staged at `1007` but blocked by the
weekly GPU quota; no replication result exists yet. X33 remains a readout
result, not a selector.

**Update 2026-10-25.** X34 v2 completed on generation seed 1007, 0.64 GPU-h.
Probe − greedy was **+0.0129** [−0.0048, +0.0343], so R-1 failed; the point
estimate is close to X33's +0.0155, but the first-half pick rose to 0.0401 and
beat the probe by −0.0103. The registered null closes the linear-selector track
(D59). X33 remains a single-seed observation, not a selector.

## D59 — X34: the midpoint readout did not replicate, and the linear-selector track closes (2026-10-25)

**Context.** X33 measured a linear midpoint readout at **+0.0155** [+0.0005,
+0.0321] over greedy on 20 stories, with a lower bound barely above zero. D58
made the first replication test the sampling draw: repeat the identical protocol
on generation seed 1007, changing nothing else.

**Evidence.** X34 v2 ran on kernel version 2, revision `7381818`, 0.64 GPU-h.
Code fingerprint, checkpoint hash and stories hash are identical to X33; the
only moved variable is the seed, verified in the pushed source.

| selector | X33 (seed 7) | X34 (seed 1007) |
|---|---|---|
| greedy | 0.0170 | 0.0170 |
| log-probability | 0.0103 | 0.0114 |
| first-half pick | 0.0277 | **0.0401** |
| **linear probe** | **0.0324** | **0.0299** |
| oracle | 0.0618 | 0.0794 |
| shuffled-label probe | 0.0205 | 0.0238 |

| contrast | X33 | X34 |
|---|---|---|
| probe − greedy | +0.0155 [+0.0005, +0.0321] | **+0.0129 [−0.0048, +0.0343]** |
| probe − first-half | +0.0047 [−0.0106, +0.0217] | **−0.0103 [−0.0350, +0.0100]** |
| probe − log-probability | +0.0222 [+0.0041, +0.0443] | +0.0185 [−0.0129, +0.0476] |
| shuffled − greedy | +0.0036 | +0.0069 |

R-1 fails: the mean clears +0.010 but the CI includes zero. R-2 passes as
registered (probe − log-probability +0.0185 ≥ 0); R-3 passes (+0.0069 < +0.010).
The runner's inherited `P2` also requires the probe to beat the half-pick, so it
prints FAIL; that is X33's stricter bar, not X34's R-2.

**Decision.**
1. **The registered verdict is a null; no selector claim.** R-1 failed, so the
   decision rule is "report a null; do not proceed to a selector."
2. **Close the linear-selector track.** X35 (a fresh story set) and the
   selective-accuracy curve were contingent on R-1; neither runs. Another
   20-story pilot would not resolve a ~0.013 effect with this variance.
3. **Report the nuance, not the label.** The point estimate moved by only
   −0.0026 from X33, so this is not evidence that the signal vanished. It is
   evidence that the effect is not resolvable on a second seed and that the
   first-half rule — now 0.0401 — is the stronger cheap baseline. The
   pre-registration's "seed artifact" wording is one explanation; "failed the
   registered replication" is the one the data support.
4. **Keep the readout result honest.** X33 stays in the record as a single-seed
   observation with this null attached. It is not a capability.
5. **Provenance follow-up.** The shared X33 runner's manifest did not record
   `SEED_BASE`, the one variable X34 moved. This run is covered by
   `docs/results/x34/provenance.json` and the pulled kernel source. The runner
   now writes `seed_base`; the archived manifest is unchanged and honest about
   what it was written by.

**Cost.** 0.64 GPU-h. Artifacts: `docs/results/x34/`; registration:
`2026-10-22-X34-seed-replication.md`.

**Status.** closed — the linear-selector track is closed. The coverage/selection
gap remains open and is the next question, not this probe.

## D60 — X35: the static code mixture does not move the 58M update (2026-10-27)

**Context.** Data is the only large positive in the record (0.169 nats at
matched tokens), and D19's 55/20/15/10 mixture had never been run. X26/X28 showed
the adaptive policy drifts to easy sources and costs +0.0183; X29's 94% delivery
path effect was confounded with corpus passes. X35 held the four published 498M
shards, the 500-step schedule, the seed and the token budget fixed, and moved
only the static source weights: control 60/20/15/5, code10 55/20/15/10, code25
40/20/15/25.

**Evidence.** Kernel `kalia-x35-mix-screen`, version 1, 7.71 GPU-h, seed 1401.

| arm | val @ 5,200 | val − control | train @ 5,200 | train − control |
|---|---|---|---|---|
| control | **2.7903** | — | 2.7511 | — |
| code10 | 2.7896 | **−0.0007** | 2.6502 | −0.1009 |
| code25 | 2.7961 | **+0.0058** | 2.2515 | **−0.4996** |

The val curves are flat across arms at every checkpoint (≤0.010 at the worst
point). **S-1 fails** (−0.0007, bar −0.010); S-2 and S-4 were contingent and did
not run; S-3 is within its +0.050 boundary; S-5 is **not measured** because the
static path writes no `per_source_log.csv`.

The code25 arm's training loss is 0.50 nats lower while its val is 0.0058 worse.
That is the X26/X28 dessert signature in aggregate: more code makes the training
mixture easier to fit, not the held-out distribution better.

**Decision.**
1. **The registered verdict is a null; no Stage 2.** Close the static-mixture
   track at the 500-step budget. Do not tune the weights until something passes;
   that is searching for a result.
2. **Do not cite this as evidence against the 0.169-nat corpus effect.** It tests
   a 5-point code shift on the equalised shards at 500 steps, not the full corpus
   change at full scale. The corpus effect remains the only large positive, and
   the next data bet must be long-form or full-scale, with a new registration.
3. **Record the dessert signal.** A lower training loss on the reweighted mixture
   is not a data win. Any future mixture arm reports train and val together.
4. **Fix the S-5 instrumentation before the next static arm.** The static path
   writes no per-source log; a standalone per-source eval at the end is the
   minimum needed for the gaming check, and claiming S-5 without it would be a
   green check that did not run.
5. **Do not spend the remaining 22.29h on a failed screen.** The next data
   question is long-form (LAMBADA/story structure) or a full-scale v0.4, not
   another code-share arm.

**Cost.** 7.71 GPU-h. Artifacts: `docs/results/x35/`; registration:
`2026-10-26-X35-static-mixture-screen.md`.

**Status.** closed — the static-mixture screen is closed; Stage 2 was not run.

## D61 — X36: the entity memory keeps names but costs language quality (2026-10-27)

**Context.** Every cheap selector failed. X33 found a midpoint signal, X34 did
not replicate it, and the linear-selector track closed. X36 tested the
architectural version: a self-supervised 32-slot entity workspace written from
the model's final states, read at the output, with a copy head and a recurrence
head, against a **parameter-matched adapter control** at 30M.

**Evidence.** Eight arms, seeds 1338/1339, 763 steps, ~5.4 GPU-h.

| arm | val mean | greedy entity | half-pick | oracle |
|---|---:|---:|---:|---:|
| control | **3.8274** | 0.000 / 0.000 | 0.008 / 0.007 | 0.019 / 0.033 |
| copy | 4.0865 | 0.008 / 0.013 | 0.066 / 0.041 | 0.109 / 0.116 |
| **memory** | 3.9033 | **0.011 / 0.016** | **0.074 / 0.090** | **0.142 / 0.174** |

- **M-1 PASS** — greedy memory − control +0.0106 / +0.0162, both seeds, bar
  0.005 (control spread 0.0).
- **M-1b PASS** — oracle far above control.
- **M-2 FAIL** — **+0.0760 nats** worse than control, allowance +0.010.
- **M-3 PASS** — recurrence AUC 0.823 / 0.824.
- **M-4 PASS** — distinct-2 memory 0.127 vs base 0.120.

**Decision.**
1. **The memory mechanism works for retention, but the registered language
   cost fails.** The pre-registered rule for "M-1 passes, M-2 fails" is: report
   the trade-off, **no scale**. No 58M run is registered from this result.
2. **The control is a strong baseline.** A parameter-matched wider MLP is 0.076
   nats better than the memory model. Capacity alone buys language quality; the
   memory spends its capacity on the workspace.
3. **The copy head alone has a retention signal** (oracle 0.109/0.116), but its
   copy accuracy is 2–3.5% and its val is worse. The workspace + recurrence adds
   the larger half/oracle gain; the greedy continuation is still 1–2%, so the
   selector problem remains.
4. **A val-preserving memory is a new registration, not a tune of this one.**
   Candidates: a much smaller workspace, a memory-only-on-entity gate, an
   auxiliary LM-preservation loss, or a calibrated copy head.
5. **Record the trade-off.** This is a component result, not a model win; the
   pre-registration's decision is binding.

**Cost.** ~5.4 GPU-h. Artifacts: `docs/results/x36/`; registration:
`2026-10-27-X36-entity-memory-micro.md`.

**Status.** closed — no scale; the memory mechanism is a component candidate,
not a model improvement.

## D62 — X37: the Pāṇinian rule grammar is a null at 30M (2026-10-28)

**Context.** After the X36 memory trade-off, X37 tested a different unit of
computation: 64 reusable low-rank rules, top-1 routing, an anuvṛtti carry, an
adhikāra scope, a general SwiGLU and an optional utsarga/apavāda exception path,
against a parameter-matched wider MLP. v1 completed base/control, then died on
the 3D-Newton–Schulz optimizer bug; v2 reran the six rule arms with the
`ndim == 2` fix and reused the valid base/control.

**Evidence.** ~14 GPU-h across both launches.

| arm | val s1338 | val s1339 | mean |
|---|---:|---:|---:|
| control (matched MLP) | **4.0692** | **4.2290** | **4.1491** |
| rules | 4.1503 | 4.2113 | 4.1808 |
| full (rules + carry + scope) | 4.1721 | 4.2564 | **4.2143** |
| full + exception | 4.1932 | 4.3872 | 4.2902 |

S-1 fails (+0.0652); S-2 fails (−0.00128 / −0.00059; the full is less
swap-sensitive); S-3 passes decisively (entropy 4.157, max share 0.018,
effective **63.9 / 64**); S-5 shows the carry/scope is worse than rules alone
on val (+0.0335) and slightly better on swap (+0.0026/+0.0020).

**Decision.**
1. **The registered SUTRA micro is a null. No 58M scale.** The rule grammar as
   implemented does not beat the parameter-matched dense MLP on val or on the
   compositional swap test.
2. **The router and codebook work; the rules are not specialized.** All 64
   rules are used at ~1.6%, so the codebook is capacity, not a grammar. A
   uniform router is a design signal, not a success.
3. **The Paninian mapping is not closed as an idea; this implementation is.**
   A new registration must change a specific mechanism: a rule-diversity /
   specialization objective, a general path matched to the control's 1,408
   width, a longer 1,500-step micro, top-k routing, or role-conditioned routing.
   No tuning of the failed 763-step run.
4. **The next-gen programme continues, but with constraints.** One failed
   registered hypothesis closes that hypothesis; it does not close the search.
   The next attempt must be cheap, parameter-matched and pre-registered.
5. **The current real model win is v0.3.0's E-1.** The final eval printed
   **2.7743** (bar 2.8048), 0.0505 better than v0.2.0; E-2/E-3 are pending.

**Cost.** ~14 GPU-h. Artifacts: `docs/results/x37/`; registration:
`2026-10-27-X37-sutra-micro.md`.

**Status.** closed — no scale. The rule-grammar line is parked until a targeted
SUTRA-v2 registration exists.

## D63 — v0.3.0 clears all three final bars and becomes the release candidate (2026-10-28)

**Context.** The second-epoch re-warm completed 2,000 steps and the final
checkpoint (step 6,770) was evaluated on CPU at zero GPU quota. The first
final-eval launch printed the canonical val then died on a markdown parse bug;
the fixed launch produced E-2/E-3 without changing any number.

**Evidence.**

| bar | required | measured | verdict |
|---|---|---|---|
| E-1 canonical val | ≤ 2.8048 | **2.7743** (bpB 0.9070) | PASS |
| E-2 2k benchmark | no task worse than −2σ | all five tasks improved | PASS |
| E-3 CL-0 | ≤ 2.805875 | **2.756648** (drift −0.042227) | PASS |

E-1 is 0.0505 nats better than v0.2.0's 2.8248. E-2 deltas: ARC-Easy +1.15 pp,
HellaSwag +1.60, LAMBADA +2.40, PIQA +0.22, WinoGrande +0.55; every z is
positive. E-3 is 0.0422 lower than the v0.2.0 probe, not merely inside the
allowance.

**Decision.**
1. **v0.3.0 is the release candidate.** All three registered bars pass; the
   flag `release_candidate` is true in `v030_final.json`.
2. **This is a model win from data/schedule, not a new architecture.** The
   architecture is unchanged. The next-gen architecture search must beat this
   checkpoint, not v0.2.0, from now on.
3. **The 500-sample benchmark verdict is not rewritten.** The 2,000-sample
   comparison is a new observation against the archived v0.2.0 2k baseline.
4. **Promotion is still a separate gate.** Release candidate is not a release;
   the model card, license and release checklist still have to pass.
5. **The next architecture test is the untested half of X11.** A simple
   6-layer × 2-loop arm was already run on 2026-09-23 and was quality-neutral
   (3.5012 vs 3.4924, +0.009, one seed). The missing claim is parameter
   efficiency: 6 × 2 loops at 30M versus 12 unique layers at ~60M, equal
   effective depth. That comparison is the next registration, not a rerun of
   the neutral arm.

**Cost.** ~10 GPU-h training + CPU final eval. Artifacts:
`docs/results/v030-epoch2/`; registration: `2026-10-27-v030-second-epoch.md`.

## D64 — X38: the counterfactual residual bottleneck is the next architecture test (2026-10-28)

**Context.** The goal is a mechanism that makes the 58M model interesting
enough for a paper, not a frontier model. X36's memory kept names but cost
**+0.076 nats**; X37's rule grammar stayed alive but used all 64 rules
**uniformly**. Both are failures at the output of an unchanged residual stream.

**Decision.** Register X38, the **Counterfactual Residual Bottleneck (CRB)**:

- Each block's residual delta is decomposed into a low-rank stable component
  and a volatile complement. The sum is the standard residual, so the language
  path is invariant to the split by construction.
- A paired counterfactual loss trains the stable stream to be invariant to
  consistent renames/synonyms and sensitive to role swaps, negation, number
  and spatial changes.
- Arms: `control`, `split` (same decomposition, no CF loss), `loss` (CF loss on
  the full hidden state), `crb`; 30M, 763 steps, seeds 1338/1339.
- Bars C-1..C-6 are fixed in
  `docs/preregistrations/2026-10-28-X38-crb-micro.md` (sha `a28329ed…`).
- The 58M confirmation is conditional on C-1/C-2/C-3/C-5 and waits for quota.

**Why CRB before X11.** X11's simple 6 × 2 vs 6 × 1 arm already ran on
2026-09-23 and was neutral (3.5012 vs 3.4924, +0.009). The untested X11 claim
is parameter efficiency at equal effective depth, which needs a ~60M dense
control. CRB is cheaper, targets two measured failures, and has a held-out
behavioural probe. X11 remains queued.

**Status.** Registered and implemented; code, tests and the kernel are
committed; **no GPU spent**. The current window has 0.87 GPU-h; the 30M screen
waits for the 2026-10-31 reset.

## D65 — X38: the invariant sum is free, the counterfactual objective collapses (2026-11-01)

**Context.** X38 tested a low-rank decomposition of every block's residual
delta into a stable stream and a volatile complement, with the sum preserved
and a paired counterfactual loss on the stable stream. The question was whether
the CF signal could shape a stable representation without paying the X36
language-quality tax.

**Evidence.** Eight arms, 30M, 763 steps, seeds 1338/1339, 5.16 GPU-h training
plus the eval-only kernel.

| bar | measured | verdict |
|---|---|---|
| C-1 CRB val ≤ split + 0.005 | **+0.0019 / −0.0025** | PASS |
| C-1b split val ≤ control + 0.005 | **+0.0006 / +0.0044** | PASS |
| C-2 CRB swap ≥ split + max(spread, 0.01) | **−0.00062 / −0.00062** | FAIL |
| C-3 CRB paraphrase ≤ split − 0.01 | **−0.02268 / +0.00258** | FAIL |
| C-4 CRB retention ≥ split − 0.005 | **−0.00069 / 0.0** | PASS (floor-limited) |
| C-5 stable rank > 8, norm > 0.01, change > 1e-3 | rank **1.48 / 1.44**; norm **1319 / 1073**; change **1.49–1.52** | FAIL on rank |
| C-6 pair audit clean | hashes match; templates/names disjoint | PASS |

**Decision.**
1. **No 58M scale. Close this implementation.** `advance = false`.
2. **The invariant-sum decomposition is a component candidate, not a model
   win.** Split and control are the same model on val (C-1b), and CRB pays no
   language tax (C-1). The decomposition is free at the language level.
3. **The counterfactual objective as implemented is a null.** It trains the
   projection (relative change ~1.5) but collapses the stable stream to
   **rank ~1.5**. A rank-1 direction that separates the pair sets is enough to
   minimise the loss; the stable stream is an edit detector, not a meaning
   representation. The trainable norm grew to 1073–1319 versus the split's 72.
4. **The split did not help and may have hurt.** CRB is slightly less
   swap-sensitive than split on both seeds (−0.00062), and the paraphrase
   effect is inconsistent (worse on 1338 by 0.0227, better on 1339 by 0.0026).
   The loss-only arm reduces paraphrase distance on both seeds but also reduces
   swap sensitivity; the CF objective teaches invariance, not sensitivity, on
   the real probe.
5. **The rank bar was load-bearing.** Without C-5, the trained projection and
   the language-neutral val would have looked like a partial win.
6. **A new registration, not a tune.** Candidates: a rank/coverage regulariser
   or reconstruction head on the stable stream; a sensitivity-only or balanced
   objective; story-level pairs instead of templates. The next portfolio bets
   are Relational Algebra Attention and Predictive-Coding Depth.

**Cost.** 5.16 GPU-h training + ~1.5h eval. Artifacts: `docs/results/x38/`;
registration: `2026-10-28-X38-crb-micro.md`.

**Status.** closed — no scale. The invariant-sum decomposition remains a
component candidate; the counterfactual objective is closed as implemented.

## D66 — v0.3.1: the 30/70 v0.2.0/v0.3.0 merge is the release candidate (2026-11-01)

**Context.** The v0.3.0 second epoch's best in-training 100-batch val was
**2.7255 at step 6,500**; the final checkpoint at 6,770 scored **2.7743**. The
intermediate checkpoint was overwritten. A weight-space merge of v0.2.0 and
v0.3.0 is valid because v0.3.0 is a continuation of v0.2.0.

**Evidence.** Kaggle CPU, zero GPU.

| v0.2.0 weight | val |
|---:|---:|
| 0.0 | 2.7743 |
| 0.1 | 2.7653 |
| 0.2 | 2.7601 |
| **0.3** | **2.7579** |

The w0.0 run reproduced 2.7743 exactly. The best blend is **30% v0.2.0 / 70%
v0.3.0**, **−0.0164 nats**. The 2k benchmark improves on all five tasks vs
v0.3.0 (ARC +0.45, HellaSwag +0.45, LAMBADA +0.70, PIQA +0.38, WinoGrande
+0.95); no task regresses by >2σ. CL-0 is **2.740949**, **−0.015699** vs v0.3.0.
`release_candidate = true`.

**Decision.**
1. **v0.3.1 is the release candidate.** It is a weight-space soup, not an
   architecture result: it recovers part of the final-checkpoint regression by
   averaging a continuation's endpoints.
2. **v0.3.0 remains the base experiment.** The merge does not change the
   architecture, data or training recipe.
3. **The checkpoint must be re-materialised before release.** The merge kernel's
   cleanup deleted the binary after scoring it; the exact blend is deterministic
   from the two hash-pinned checkpoints. The release uses the materialised file
   only after the 100-batch val reproduces **2.7579**.
4. **The 2k gains are modest** (z 0.24–0.50); a 5k re-score is the confirmation
   before a public benchmark claim.
5. **Next:** export v0.3.1 to safetensors and re-score the written file (the
   v0.2.0 release gate); update the model card. Optional: w0.4/w0.5 or a
   lower-LR best-checkpoint continuation. The data mix waits.

**Cost.** Zero GPU; ~3 CPU-hours. Artifacts: `docs/results/v031-merge/`;
plan: `docs/research/2026-11-01-v031-merge-plan.md`.

**Status.** released public at `kalia-lm/kalia-v031` on 2026-11-01; 15 files;
the old v0.1.2/v0.2.0 cards are untouched.

**Update 2026-11-01.** The materialise kernel rebuilt w0.3 from the hash-pinned
sources and re-scored **2.7579 / bpB 0.9016**; checkpoint sha
`f59e059c…`, weight sha `19f68b…`, 668,830,321 bytes. The export kernel then
wrote 93 tensors and re-scored the written `model.safetensors` at
**2.7579 / bpB 0.9016** (334,360,208 bytes, sha `bbd8a036…`). The release is
published publicly with the card and three static charts; the old cards remain
frozen.

## D67 — X40 build: factual prompt, counterfactual target, fixed weights (2026-11-01)

**Context.** The X40 registration fixed the mechanism and the I-1…I-7 bars but
left the implementation open. The first pair generator rendered the **edited**
prefix into the model input, which would have made the intervention already
applied and turned I-3 into ordinary continuation. The target also had two
possible readings: the factual boundary state or the edited-prefix state.

**Decision.**
1. The model sees the **factual** prefix and an **explicit** instruction
   (`rename Maya to Tessa`, `change two to three`). The edited prefix is never
   shown; it is only the input to the detached next-state target.
2. The target is the edited prefix's **last-token trunk state**, before the
   workspace read, detached.
3. The 30M weights are fixed before GPU: pair LM **1.0**, state **0.1**,
   diversity **0.01**, pair batch **8**, every step. The 30M pairs are 16,000
   train / 1,050 eval.
4. The workspace is 16 slots × 64, gate init **0.0**, zero-init read.

**Why.** The explicit instruction is required for `rename`/`number`/`place` to
be well-posed. The edited-prefix target is the only one that forces a
counterfactual representation. An open write gate with a zero read keeps the
model identical to control at init while letting the state loss train the
workspace immediately; X36's closed gate is the negative precedent.

**Cost.** Zero GPU; 644 tests passed after the build.

**Status.** Built and CPU-tested; the 30M screen is run and failed — see D68. Artifacts:
`intervention_pairs.py`, `eval_ip.py`, `tools/kalia_x40_ip_kernel.py`,
`docs/research/2026-11-01-x40-build.md`.

**Update 2026-11-01.** The code dataset was republished with the new files and
the 30M screen was pushed as `subhajitlucky/kalia-x40-ip`, revision
`9588b85e923b`. The kernel is running; no I-bar is measured yet.

## D68 — X40 is closed at 30M: no composition gain, and a language tax (2026-11-02)

**Context.** The X40 30M screen completed as kernel version 2 at revision
`2153d6f`. The v1 kernel died at the first workspace arm on an fp16
`masked_fill(-1e9)` overflow and produced no eval; it is logged separately and
is not used. v2 ran all five arms at seeds 1338/1339 (763 steps), with the
registered weights and no changes after launch.

**Result.**

| bar | 1338 | 1339 | verdict |
|---|---:|---:|---|
| I-1 IP val − control | **+0.2040** | **+0.1689** | **FAIL** |
| I-2 IP seen − control | +0.2022 | +0.1867 | PASS |
| I-3 IP unseen − pairs | **+0.0283** | **0.0000** | **FAIL** |
| I-4 IP retention − control | −0.00128 | −0.00036 | PASS (floor) |
| I-5 entropy / rank / probe | 3.79 / 12.28 / 0.889 | 3.76 / 11.96 / 0.889 | PASS |
| I-6 pair audit | clean | clean | PASS |
| I-7 control seed spread | val 0.0113 / seen 0.0244 / unseen 0.0100 | — | reported |

`advance = false`. The IP seen accuracy is only 0.536/0.544, 3.6–4.4 pp above
2-way chance; unseen is 0.265/0.302, 1.5–5.2 pp above 4-way chance. The
workspace is alive, so the null is not a dead-module artifact.

**The tax is not only the state loss.** Pairs-only costs **+0.123/+0.122** nats
vs control; the event workspace adds **+0.094/+0.054** over pairs; the
state/diversity terms **slightly reduce** the tax vs the workspace arm
(**−0.014/−0.008**). The pair data and the workspace architecture are the
costs; the state term is not the culprit. The total is still 34–41× the 0.005
allowance.

**The seen gain is vs control, not vs pairs.** Against the pair-only baseline,
the IP seen gain is **+0.193 on seed 1338 but +0.004 on seed 1339**; the
`pairs` arm itself is unstable (0.342 vs 0.540). I-2 passes as registered, but
the IP does not robustly beat the data baseline on the seen task.

**The parameter-matched adapter is the surprise.** The output-only adapter
(no pair data) improves val by **0.26 nats** vs control; the workspace at the
same budget is **0.44–0.48 nats worse than the adapter** (and **0.18–0.22 nats
worse than control**). At 30M the extra parameters are better spent at the
output than in the event workspace. This is a capacity observation, not a
rescue for X40.

**The composition test has a confound.** `number+place` is 0.0 in 9/10 arms
(pairs-s1339 0.107): the `pack_box` eval template's causal relation
("packed" → "bag holds") never appears in training. Excluding it, the other
three compositions still do not give a consistent +0.10: rename+swap
+0.067/+0.107, swap+negate +0.027/−0.020, rename+negate +0.020/+0.020. The
confound changes the size of the composition claim, not the verdict.

**Decision.**
1. **Close X40.** No 58M, no 134M. No eval tuning, no pair-weight rescue, no
   composition-set rescue. The registration's rule is I-1 or I-3 fail → stop.
2. **v0.3.1 remains the model.** The architecture/objective search does not
   have a promoted mechanism.
3. **The next mechanism needs a new pre-registration.** A cleaner composition
   benchmark should hold the causal relation fixed across train/eval and vary
   only the composition; that is a benchmark contribution, not a scale-up of
   X40.
4. **The adapter result is a side observation**, not a new extraordinary
   mechanism. If anyone pursues it, it is a separate pre-registered maintenance
   run, not a rescue.

**Cost.** v1 ~2.2 GPU-h before the startup failure; v2 ~6.3 GPU-h. No further
GPU. Artifacts: `docs/results/x40/` (26 files; summary, rows, manifest, pair
audit, configs, val reports, log). Registry: the X40 row lists 26 files; the current registry is 35 runs / 260 files.

## D69 — The clean 2×2 benchmark confirms the composition failure (2026-11-03)

**Context.** X40's unseen-composition set mixed a new operation pair with a new
sentence type / causal relation (`pack_box`). To test whether the null was a
test artifact, a clean 2×2 benchmark was built: seen/unseen sentence type ×
single/pair change, one new name pool held constant, degenerate compositions
removed. The archived X40 v2 checkpoints (control, pairs, IP at seeds
1338/1339) were scored on it by a CPU kernel. This is post-hoc and does not
re-score the registered I-1…I-7.

**The 2×2** (chance 0.50 for single cells, 0.25 for pair cells):

| cell | control 1338/1339 | pairs 1338/1339 | IP 1338/1339 |
|---|---:|---:|---:|
| S-S seen / single | 0.4375 / 0.4300 | 0.6350 / 0.7200 | 0.7050 / 0.7200 |
| **S-P seen / pair** | **0.2400 / 0.2400** | **0.2525 / 0.2900** | **0.2750 / 0.2275** |
| U-S unseen / single | 0.3650 / 0.4000 | 0.3700 / 0.5425 | 0.5400 / 0.5550 |
| U-P unseen / pair | 0.3300 / 0.3450 | 0.3325 / 0.3425 | 0.3275 / 0.3275 |

**S-P by operation** (chance 0.25):

| operation | control 1338/1339 | pairs 1338/1339 | IP 1338/1339 |
|---|---:|---:|---:|
| `number+place` | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| `rename+swap` | 0.239 / 0.179 | 0.246 / 0.269 | 0.239 / 0.246 |
| `swap+negate` | 0.236 / 0.326 | 0.292 / 0.348 | 0.382 / 0.270 |
| `rename+negate` | 0.323 / 0.323 | 0.316 / 0.368 | 0.331 / 0.256 |

**Findings.**
1. The X40 null is **not** a template artifact. On the cleanest cell (S-P),
   every arm is at chance even though the sentence type and causal relation
   were seen in training.
2. `number+place` is **0.0 for every model**, even on the seen `put_count`
   template. The model saw `number` and `place` singly; it cannot combine them.
3. `rename+swap` is at chance for every model.
4. The above-chance pair cells are negation-marked; the **control** is already
   at 0.32–0.33. That is a “did not”/new-name heuristic, not a learned
   composition. The IP does not consistently improve these cells.
5. The IP/pairs gains are on the **single** operations and the easy negation
   pairs. They do not improve the hard compositions.
6. Aggregate composition accuracy hides the effect: a 4-way aggregate can look
   “at chance” because one hard operation is 0.0 while another is 0.4. Future
   claims must report by operation.

**Decision.**
1. **X40 stays closed.** No 58M, no 134M, no re-score of the registered bars.
   The clean benchmark confirms the failure is composition, not the test.
2. **The benchmark is the reusable asset.** It separates template novelty from
   composition and exposes the hard operations.
3. **A future mechanism must target `rename+swap` and `number+place`.** Beating
   the negation pairs or the single-op cells is not enough; the control already
   handles those. That is a new pre-registration, not a rescue.

**Cost.** CPU only; no GPU. Artifacts: `docs/results/x40-bench/` (summary,
9,600 per-record rows, audit, kernel log). Registry updated separately.

## D70 — Gold-edit test: the edit is not the missing piece (2026-11-03)

**Context.** The clean 2×2 benchmark left one question: is the model failing to
*apply* the intervention, or failing to *continue* the intervened world? The
gold-edit mode gives the model the post-intervention prefix (no instruction)
and scores the same candidates. Same archived control/pairs/IP checkpoints, CPU
only.

**S-P aggregate, normal → edited** (chance 0.25):

| model | seed 1338 | seed 1339 |
|---|---:|---:|
| control | 0.240 → 0.293 | 0.240 → 0.295 |
| pairs | 0.253 → 0.255 | 0.290 → 0.260 |
| IP | 0.275 → 0.320 | 0.228 → 0.325 |

**S-P by operation, normal → edited:**

| operation | control | pairs | IP |
|---|---|---|---|
| `number+place` | 0.000 → 0.000 / 0.000 → 0.000 | 0.000 → 0.000 / 0.000 → 0.000 | 0.000 → 0.318 / 0.000 → 0.000 |
| `rename+swap` | 0.239 → 0.254 / 0.179 → 0.216 | 0.246 → 0.239 / 0.269 → 0.261 | 0.239 → 0.239 / 0.246 → 0.269 |
| `swap+negate` | 0.236 → 0.180 / 0.326 → 0.225 | 0.292 → 0.157 / 0.348 → 0.247 | 0.382 → 0.303 / 0.270 → 0.348 |
| `rename+negate` | 0.323 → **0.504** / 0.323 → **0.519** | 0.316 → 0.421 / 0.368 → 0.353 | 0.331 → 0.414 / 0.256 → **0.474** |

**Findings.**
1. The gold edit does **not** unlock the hard cells: `number+place` is 0.0 in
   5/6 arms (the lone IP-s1338 0.318 does not replicate), and `rename+swap` is
   at chance in every arm.
2. The S-P aggregate rise is mostly `rename+negate`, which the **control** uses
   best (0.504/0.519 vs IP 0.414/0.474). The model can continue some
   negation-marked pairs from the edited world; that is not the composition the
   mechanism was meant to teach.
3. The IP’s gold-edit S-P is +0.065 over pairs on both seeds (0.320/0.325 vs
   0.255/0.260), consistent but small; the control is better on the largest
   cell. This is not a general composition ability.
4. The gold-edit prompt is zero-shot and out of distribution; the models were
   never trained to continue from it. A fine-tuned edit interface could score
   higher. The hard-cell failure is still informative: continuing from the
   edited world should be easier than applying the edit, and the hard cells
   remain at 0/chance.

**Decision.**
1. **Stop the composition line.** No edit-interface bet on this evidence; the
   edit is not the missing piece. The hard cells are role binding
   (`rename+swap`) and two independent attribute changes (`number+place`).
2. **X40 stays closed.** No 58M/134M. The benchmark + negative record are the
   artifact; v0.3.1 remains the model.
3. **A future bet, if any, must target binding/attribute composition**, with
   the S-P by-operation table as the bar; the gold edit is not enough.

**Cost.** CPU only; no GPU. Artifacts: the gold-edit summary/rows/log under
`docs/results/x40-bench/`.

## D71 — V032 output adapter passes the val bar; 2k guard pending (2026-11-04)

**Context.** The audit found no A-class mechanism. The only positive val signal
was the X40 parameter-matched output adapter. V032 tests it on v0.3.1: freeze
the base, train only a zero-init output adapter (`512 → 1024 → 512`,
**1,048,576** params) for 2,000 steps, two seeds, canonical 100-batch val.

**Result.**

| seed | val | base val | delta |
|---|---:|---:|---:|
| 1401 | **2.7493** | 2.7580 | **−0.0087** |
| 1402 | **2.7495** | 2.7580 | **−0.0085** |

- Bar: adapter ≤ base − 0.005, both seeds. **Pass.**
- Frozen-weight checksum unchanged, both seeds. **Pass.**
- Adapter parameter count exact. **Pass.**
- Base published val 2.7579; re-measured 2.7580.
- Kernel `subhajitlucky/kalia-v032-adapter` v2, revision `c40b464`; ~3.4 GPU-h.

**Decision.**
1. **Val bar passes; V032 is an internal v0.3.2 candidate.**
2. **The 2,000-sample benchmark pair is the conditional guard (V-5).** If any
   task regresses beyond 2σ vs v0.3.1, withhold the candidate. If not, record a
   small val improvement.
3. **This is engineering, not a new mechanism.** It does not answer the
   composition failure; it is a known capacity/placement result.
4. No tuning of LR, steps or adapter size after the result.

**Artifacts.** `docs/results/v032-adapter/` (summary, logs, adapter state
dicts, kernel log). Pre-registration hash `4ead53b6…`; ledger 34 entries.

**Update 2026-11-05 — V-5 passed; the 2k is neutral.** The paired 2k guard ran
base v0.3.1 and both adapters in one CPU kernel. The fresh base reproduced the
published 2k numbers exactly. No task regressed beyond 2σ: ARC +0.45/+0.60 pp,
PIQA −0.22/−0.11 pp, HellaSwag +0.05/−0.35 pp, LAMBADA acc −0.85/−0.85 pp,
LAMBADA ppl +4.8/+6.3 (σ ~31), WinoGrande −0.08/+0.08 pp.

**Decision.** V032 is a **val-only internal candidate**: val −0.0087/−0.0085,
2k neutral. It is not a public release, not a benchmark improvement, and not a
new mechanism. The public model remains v0.3.1.

## D72 — X41 edit compiler: partial advancement (2026-11-05)

**Context.** The full 2×2 baseline on v0.3.1 (58M) showed the reader is
competent in the edited world: normal S-P **0.170** vs gold-edit **0.5125**;
`number+place` **0.000 → 0.694**; `rename+swap` **0.187 → 0.561**. The
bottleneck is edit application, not continuation. X41 trained a rank-16 LoRA
edit compiler (1,413,120 params, base frozen) to write the post-intervention
prefix, then let the frozen reader score candidates.

**Result (Arm B, held-out pair types).**

| seed | singles EM | compiler EM | S-P | `number+place` | `rename+swap` | U-P |
|---|---:|---:|---:|---:|---:|---:|
| 1401 | 0.9966 | 0.4263 | **0.4275** | **0.6935** | **0.3537** | 0.5675 |
| 1402 | 0.9920 | 0.4000 | **0.4100** | **0.6935** | **0.4830** | 0.6350 |

Arm A (singles only, seed 1401): S-P 0.4075, `number+place` 0.6935,
`rename+swap` 0.3844, U-P 0.6325.

- **C-1 PASS** (singles ≥0.90), **C-2 PASS** (`number+place` ≥0.40 both),
  **C-4 PASS** (S-P ≥0.40 both), **C-5 PASS** (U-P ≥0.30 both),
  **C-6 PASS** (base frozen).
- **C-3 FAIL**: `rename+swap` 0.3537 on seed 1401, below the 0.45 bar;
  seed 1402 passes at 0.4830.

**Decision.**
1. **X41 is a partial advancement, not a full registered pass.** The strict
   rule requires C-2 and C-3 on both seeds; C-3 misses on one. Do not
   cherry-pick seed 1402.
2. **The clean positive result is `number+place`:** 0.000 → **0.6935** on both
   seeds, exactly the gold-edit ceiling. S-P and U-P also improve materially.
3. **`rename+swap` is the remaining failure**, with a real seed spread.
4. The public base model remains **v0.3.1**; the adapters are experimental
   artifacts, not a release.
5. X42 tests a reader-guided ensemble; X43 tests template-held-out pair
   training. Both are separate registrations.

**Artifacts.** `docs/results/x41-edit-compiler/` (baseline, adapters, logs,
generated prefixes, end-to-end rows). Pre-registration hash `d3d5dc53…`.

## D73 — X42 reader-guided ensemble: null (2026-11-05)

**Context.** X41's two Arm B seeds disagree on `rename+swap` (0.354/0.483).
X42 used the frozen reader's candidate distribution to select between their
edits, with a rule chosen on a deterministic dev half and evaluated once on the
held-out test half.

**Result.** Chosen rule: MinEntropy. Test half (n=829): S-P **0.4144**,
`number+place` **0.6143**, `rename+swap` **0.4286** (bar 0.45), U-P 0.6186.
Best single seed on the same half: B_1402 `rename+swap` 0.4898; oracle
selection also 0.4898.

**Decision.** **Null.** The reader cannot repair the compiler's
`rename+swap` distribution; the oracle is capped by the better single seed.
No selector tuning after the test. X43 tests data coverage.

**Artifacts.** `docs/results/x42-reader-ensemble/`. Pre-registration hash
`0746ac10…`.

## D74 — X43 template-held-out pair training (2026-11-05)

**Context.** X41's compiler never applied `rename` then `swap`. X43 teaches
all four test pair types on the three extra templates (`carry_place`,
`give_place`, `count_place`), never on the benchmark templates, then tests on
the benchmark. This is a template-transfer test, not the stricter
held-out-operation-combination test of X41.

**Status.** Registered and running on Kaggle GPU (`kalia-x43-thpt`), two
seeds, same rank-16 LoRA. Bars D-1…D-6: singles EM ≥0.90, `number+place`
≥0.40, `rename+swap` ≥0.45, S-P ≥0.45, U-P ≥0.35, base frozen. Cost cap
2 GPU-h. Result pending; no tuning after the result.

**Update 2026-11-05 — D-1…D-6 pass.** X43 trained the same rank-16 LoRA on
the X41 Arm B data plus all four test pair types on the three extra templates
only. The benchmark templates never received pair supervision.

| seed | singles EM | compiler EM | S-P | `number+place` | `rename+swap` | U-P |
|---|---:|---:|---:|---:|---:|---:|
| 1401 | 0.9920 | 0.5469 | **0.4800** | **0.6935** | **0.5918** | **0.8600** |
| 1402 | 0.9977 | 0.5438 | **0.4925** | **0.6935** | **0.6463** | **0.8725** |

- **D-1…D-6 all PASS.** Cost **0.97 GPU-h**.
- `rename+swap` 0.187 → **0.592/0.646**; S-P 0.170 → **0.480/0.493**;
  `number+place` 0.000 → **0.694**; U-P 0.093 → **0.860/0.873**.
- This is a **template-held-out pair-transfer** result: pair types are trained
  on disjoint templates, not held out entirely. X41 remains the stricter,
  partial held-out-combination result.

**Decision.** X43 is the first clean positive capability result. Release the
two adapters as the **v0.4.0-x43-edit-compiler** capability pack. The public
base model remains **v0.3.1**; the standard five-task benchmark is unchanged.
The release note is `docs/releases/v040-x43-edit-compiler.md`. No tuning after
the result.

## D75 — X44/X45 fresh benchmark: boundary result (2026-11-05)

**Context.** X43 passed its registered 2×2 bars, but that benchmark had been
used across X41–X43. X44 evaluated the same adapters on new names and new
sentence templates; X45 tested zero-shot three-operation composition.

**Result (0.24 GPU-h).**

| cell | normal | gold | C_1401 | C_1402 |
|---|---:|---:|---:|---:|
| F-S | 0.3875 | 0.7125 | 0.7275 | 0.6950 |
| F-P | **0.0850** | 0.5725 | **0.5225** | **0.4925** |
| F-T | **0.0350** | 0.3925 | **0.2825** | **0.3650** |

- **F-1 PASS** (fresh pair ≥0.40), **F-2 PASS** (`number+place` 0.4545 both),
  **F-4/F-5 PASS**.
- **F-3 FAIL**: fresh `rename+swap` **0.4059/0.4158**, below 0.45.
- **F-6 FAIL**: fresh 3-op **0.2825/0.3650**; one seed below 0.30.
- Compiler exact match on fresh templates drops to **0.1850/0.1242**.

**Decision.** X44 is a **partial generalization**: pair composition and
`number+place` transfer to fresh templates, but `rename+swap` is the boundary
and 3-op depth is not seed-stable. The X43 release remains scoped to its
registered benchmark. No v0.4.1/v0.5.0 broad claim. A robust OOD compiler
(X46) would need broader template diversity; X44 stays as the held-out test.

**Artifacts.** `docs/results/x44-fresh/`. Pre-registration hash `45b516f2…`.

## D76 — HF release gate: +10% vs v0.3.1 (2026-11-05)

**User rule.** A new model is released on Hugging Face once it surpasses the
v0.3.1 benchmark by 10%.

**Definition used here.** Standard five-task zero-shot average at 2,000
samples, **relative +10%** vs v0.3.1 (mean ≈ 0.4431 → ≥0.4874), with no task
regressing beyond 2σ. This is the honest base-model metric.

**Current state.** The standard five-task suite is unchanged; the edit
compiler is a separate adapter and does not move it. The composition benchmark
is far above +10% (S-P 0.170 → 0.480/0.493; `number+place` 0.000 → 0.694;
`rename+swap` 0.187 → 0.592/0.646), but that is a capability adapter, not a
new base model.

**Decision.** No HF base-model release now. A +10% standard gain needs a new
pretraining run, not the remaining ~10 GPU-hours. Reserve the quota for a
pre-registered base run in the next window. If the user means the composition
metric, prepare a **merged capability checkpoint** separately and label it as
a capability release, not a new base model. Do not claim a standard-benchmark
gain that was not measured.

## D77 — X46 broad-template compiler: clean OOD pass (2026-11-05)

**Context.** X44 found the fresh-template boundary: F-P 0.523/0.493,
`rename+swap` 0.406/0.416, 3-op 0.283/0.365, compiler EM 0.185/0.124. The
hypothesis was data diversity, not architecture.

**Treatment.** Same rank-16 LoRA, trained on 17 templates (X43's seven plus
ten new broad templates), all 1/2/3-op combinations. The four X44 test
templates were never seen. Two seeds, 3,000 steps, **0.96 GPU-h**.

**Result (untouched X44 fresh benchmark).**

| seed | F-S | F-P | F-T | `number+place` | `rename+swap` | compiler EM |
|---|---:|---:|---:|---:|---:|---:|
| 1401 | 0.7475 | **0.6825** | **0.6150** | **0.4545** | **0.7228** | **0.3550** |
| 1402 | 0.7525 | **0.6850** | **0.5575** | **0.4545** | **0.6634** | **0.3783** |

- **G-1…G-6 all PASS.** Fallback 0.0000; base frozen.
- Fresh F-P 0.085 → **0.683/0.685**; `rename+swap` 0.406/0.416 →
  **0.723/0.663**; 3-op 0.283/0.365 → **0.615/0.558**.
- Broad template diversity closed the boundary. The X44 templates/names were
  never in training.

**Decision.** X46 is a robust OOD capability. Release **v0.4.1 — X46
broad-template compiler** on GitHub. The public base remains **v0.3.1**; the
standard five-task suite is unchanged. No HF base-model release: the standard
+10% gate is not met. If the user means the composition metric, this is a
capability checkpoint candidate, not a new base model.

## D78 — Versioning: stay on v0.x until the user declares v1 (2026-11-05)

**Rule.** No v1/v2 labels are assigned by the project. The line stays **v0.x**
until the user explicitly says a checkpoint is KALIA v1.

**Working scheme.**

- **v0.3.1** — current public base.
- **v0.4.x** — edit-compiler capability packs (v0.4.0 X43, v0.4.1 X46).
- **v0.5.x** — K-Tok tokenizer experiments.
- **v0.6.x** — K-Attn attention experiments.
- **v0.7.x** — combined from-scratch base (K-Tok + winning K-Attn).
- **v1.0** — only by explicit user declaration.

Model-size scaling is a separate decision from version numbering. The next
base experiments remain 58M until the evidence says capacity-limited.

## D79 — Owned-stack roadmap: K-Tok first, then K-Attn (2026-11-05)

**Goal.** Remove the remaining borrowed components: the GPT-2 BPE tokenizer
and any pretrained weights. No distillation, no copied inference. Standard
mathematical components and PyTorch are tools, not borrowed models.

**Sequence.**
1. **K-Tok v0** — byte-level BPE at 32k/50k/64k on decoded corpus text;
   measure compression, round-trip, domain behavior, and tied-embedding budget
   against GPT-2 BPE.
2. **K-Attn** — two prior-art probes, three candidates, matched micro-ablation
   at 30M/58M. If none wins, keep QK-Norm/RoPE and record the null.
3. **v0.7.x** — from-scratch 58M with K-Tok + winning K-Attn, compared on the
   9-task suite.
4. Scale to 134M only if v0.7.x is capacity-limited.

**Versioning.** Stay v0.x until the user explicitly declares v1 (D78).

**Artifacts.** `docs/research/2026-11-05-owned-stack-roadmap.md`.

## D80 — K-Tok v0: own tokenizer candidate (2026-11-05)

**Context.** The model still borrowed GPT-2's 50,257-token BPE. At 58M, the
tied embedding is 25.73M parameters — nearly half the model.

**Result.** Four byte-level BPE tokenizers trained on 50M decoded corpus
tokens; 2M held-out tokens.

| tokenizer | vocab | English B/tok | code B/tok | numbers B/tok | tied emb | vs GPT-2 |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 | 50,257 | 4.319 | 2.230 | **2.478** | 25.73M | — |
| **K-Tok 32k** | 32,000 | **4.380** | **3.162** | 2.152 | **16.38M** | **+9.35M** |
| K-Tok 50k | 50,000 | 4.459 | 3.262 | 2.320 | 25.60M | +0.13M |
| K-Tok 64k | 64,000 | 4.493 | 3.320 | 2.411 | 32.77M | −7.04M |
| K-Tok 32k digits | 32,000 | 4.297 | 3.114 | 1.049 | 16.38M | +9.35M |

- **32k beats GPT-2 on English/code compression** and frees 9.35M parameters.
- **GPT-2 wins on numbers**; digit-atomic tokenization is much worse for
  compression (1.049 B/tok). Not the general-text candidate.
- 50k/64k are not worth the embedding cost.
- Round-trip is **lossless** for all four; the original `false` flags were a
  `skip_special_tokens=True` measurement bug. Verified in `ktok_verify.json`.

**Decision.** K-Tok 32k is the first owned-tokenizer candidate. Next: a
matched tokenizer A/B — GPT-2 BPE vs K-Tok 32k — at 30M (2 seeds), then the
winning recipe at 58M. No model change now; v0.3.1 remains public.

**Artifacts.** `docs/results/ktok-v0/`.

## D81 — X47 Stage 0: the standard suite and the weak spots (2026-11-05)

**Result (v0.3.1, 2k where available).** ARC-Easy 0.4800, ARC-Challenge
0.2014, BoolQ 0.5435, HellaSwag 0.3235, LAMBADA 0.2700, OpenBookQA 0.1640,
PIQA 0.6197, SciQ 0.7080, WinoGrande 0.5225. **Mean 0.4258.**

**Weakest four:** OpenBookQA 0.164, ARC-Challenge 0.201, LAMBADA 0.270,
HellaSwag 0.324. These are knowledge, reasoning, and long-range recall — not
edit application. The 30-hour base-data experiment targets these without
regressing the strong five.

## D82 — X48 matched data ready; tokenizer A/B begins (2026-11-05)

Same 312M GPT-2 tokens decoded and re-tokenized with K-Tok 32k: 309.3M K-Tok
tokens, 0.8% fewer. Both arms see the same text. First 30M arm (GPT-2, seed
1338) running in the current GPU window; the remaining three arms run in the
30-hour window. Evaluation is a separate CPU kernel. Pre-registration hash
`0c7a054c…`; bars H-1…H-5.

## D83 — X48 GPT-2 arm: training and 9-task baseline (2026-11-06)

**Training.** 30M parameters (6 layers, 384 embd, 6 heads, context 512),
4,770 steps, 312M GPT-2 tokens. Final val loss **3.2021** nats/token at step
4750; checkpoint sha `ac9f1a94…`; 359MB including optimizer.

**9-task eval (2k where available).** SciQ 0.5810, PIQA 0.5686, WinoGrande
0.5217, BoolQ 0.4690, ARC-Easy 0.3920, HellaSwag 0.3010, ARC-Challenge
0.1741, OpenBookQA 0.1560, LAMBADA 0.1275. **Mean 0.3657.**

This is the GPT-2 BPE baseline for X48. The K-Tok 32k arm is still training;
its eval uses the K-Tok tokenizer. The 30M screen is a tokenizer A/B, not a
release; the 58M base remains v0.3.1.

## D84 — X48 K-Tok arm trained (2026-11-06)

**Training.** 30M parameters, 4,770 steps, 309.3M K-Tok tokens (same text as
the GPT-2 arm). Final val loss **3.3706** nats/token; checkpoint sha
`07a12913…`; 275MB including optimizer. Throughput ~39k tok/s vs ~26k for the
GPT-2 arm; the K-Tok model is smaller (32k tied embedding).

**Val caveat.** Per-token loss is not comparable across tokenizers. Over the
same text: GPT-2 ~32.02M total nats, K-Tok ~33.45M; GPT-2 is ~4.5% better on
val bpB. The 9-task suite is the deciding metric.

**Status.** K-Tok-tokenizer 9-task eval running on CPU. The first matched pair
(GPT-2 vs K-Tok, seed 1338) will decide whether the tokenizer line continues
to 58M or is parked.

## D85 — X48 tokenizer A/B: GPT-2 BPE wins; K-Tok parked (2026-11-06)

**Result (30M, seed 1338, 9-task 2k).** GPT-2 mean **0.3657**; K-Tok 32k
mean **0.3629**, Δ **−0.28 pp**. LAMBADA improves +3.85 pp (~3.4σ), but
OpenBookQA −2.80 and ARC-Challenge −1.45 lose more. Val bpB is ~4.5% worse
for K-Tok over the same text. K-Tok is smaller (275MB vs 359MB checkpoint)
and faster (39k vs 26k tok/s), but not a benchmark win.

- **H-1 FAIL**, **H-2 PASS**, **H-3 FAIL**, **H-4/H-5 PASS**.
- The screen stopped after seed 1338: the first matched pair is already below
  the +1.0 pp bar and val bpB is worse; a second seed cannot rescue the claim.

**Decision.** Keep GPT-2 BPE. Park K-Tok 32k. Do not spend a 58M run on this
tokenizer. The owned-tokenizer goal remains open for a different design later.
The next 30-hour window goes to the reasoning-heavy data experiment on
OpenBookQA, ARC-Challenge, LAMBADA, HellaSwag.

**Artifacts.** `docs/results/x48-tokenizer-ab/`. Pre-registration hash
`0c7a054c…`.

## D86 — X49 reasoning-heavy data ready (2026-11-06)

**Data.** Two 524M-token mixtures from the same source shards:
control 60/20/15/5 FineWeb-Edu/TinyStories/Cosmopedia/code; treatment
45/5/40/10. Both 1.048GB, 10M val tokens; train shas `dfa712d9…` and
`6cb98123…`. CPU only.

**Status.** 50-step smoke test running to validate the v0.3.1 resume and
re-warm pipeline. The two full 58M arms run in the 30-hour window; K-1…K-4
pre-registered (hash `3e86a1a3…`).

## D87 — X49 smoke passes; runtime amended to 1,500 steps (2026-11-06)

**Smoke.** 50 steps on the control mixture from v0.3.1; val 2.8085; throughput
**13.6k tokens/s**; checkpoint saved. The resume/re-warm pipeline works.

**Runtime.** The registered 2,000-step arm is 524M tokens ≈ 10.7h, over
Kaggle's 9-hour session limit. Amendment 1 sets both arms to **1,500 steps**
(393M tokens), warmup 150, one session each; bars K-1…K-4 unchanged.

**Status.** Full control/treatment arms run in the 30-hour window. No
result-driven tuning.

## D88 — KALIA-Vision v0 data ready (2026-11-06)

**Data.** 50k/5k/5k synthetic 32×32 images with shape/color/position captions;
fully generated by `synthetic_vision.py`; no external images or pretrained
vision weights.

**Status.** 500-step conditional-DDPM smoke running. Full 20k-step model in the
30-hour window; bars W-1…W-3 registered (shape ≥0.80, color ≥0.80, position
≥0.70). The public base remains v0.3.1; this is a capability module.

## D89 — KALIA-Vision v0 passes all bars (2026-11-06)

**Result.** Frozen KALIA v0.3.1 text encoder + 3.75M-parameter conditional
DDPM trained from scratch on 50k synthetic 32×32 shapes. 20,000 steps,
40.0 minutes. On 5,000 held-out captions:

- **W-1 shape 0.9472** (bar 0.80)
- **W-2 color 0.9684** (bar 0.80)
- **W-3 position 0.8524** (bar 0.70)
- **W-4 pass**: no pretrained vision weights; KALIA frozen.
- **W-5 pass**: sample grid saved.

Real-test classifier sanity: shape 1.000, color 1.000, position 0.957.

**Decision.** Release **v0.5.0-vision** as a capability module. The public
base remains v0.3.1. This is a synthetic 32×32 toy domain, not a general
text-to-image model; the release note says so. Next: KALIA-Audio v0
(from-scratch speech-command encoder feeding KALIA, then TTS).

**Artifacts.** `docs/results/kalia-vision-v0/`. Pre-registration hash
`0c04af74…`.

## D90 — KALIA-Vision v1 two-object scenes passes (2026-11-06)

**Result.** Frozen KALIA v0.3.1 text encoder + 3.75M conditional DDPM from
scratch on 50k synthetic two-object 32×32 scenes. 25,000 steps, 45.5 minutes.
On 5,000 held-out captions:

- left shape **0.9910**, left color **0.9804**
- right shape **0.9932**, right color **0.9756**
- exact two-object scene **0.9482**
- real-test classifier sanity: exact 0.9994
- V1-1…V1-6 pass.

**Decision.** Release **v0.5.1-vision-two** capability update. Public base
remains v0.3.1. Caveat unchanged: synthetic 32×32 toy domain, not general
text-to-image. Next: KALIA-Audio v0 (from-scratch speech-command encoder feeding
KALIA, then TTS).

## D91 — KALIA-Audio v0 passes (2026-11-06)

**Result.** From-scratch 136k-parameter 1D CNN on six synthetic formant
commands. 15 epochs, 0.38 minutes. Held-out test **0.912**, val 0.916; all six
commands present; class 5 weakest at 0.766.

**Decision.** Release **v0.6.0-audio-commands** capability pack. Public base
remains v0.3.1. Caveat: synthetic formant tones, not real speech. Next: real
speech commands, then KALIA response generation and TTS.

## D92 — KALIA-Audio v1.1 real speech passes (2026-11-07)

**Result.** Log-magnitude STFT + 2D CNN on 12 real Speech Commands words,
10,800 train / 4,891 official test. 391,884 params, 30 epochs, 5.7 minutes.
Test **0.8561**, val **0.8517**; R-1…R-3 pass. v1 raw-waveform baseline was
0.655.

**Decision.** Release **v0.6.1-audio-speech** capability update. Public base
remains v0.3.1. Caveat: 12 words, not open-vocabulary speech. Next: 35-word
vocabulary, KALIA response generation, TTS.

## D93 — X49 data mixture: near-miss, parked (2026-11-08)

**Result (9-task, 2k).** Control mean **0.4198**, treatment **0.4280**,
Δ **+0.82 pp**. BoolQ **+5.45 pp** (3.45σ), WinoGrande **+1.89 pp**; the
targeted weak-task average is **−0.10 pp**. Shared control val **2.80249 →
2.80061** (−0.0019).

- **K-1 FAIL** (+0.82 < +1.0).
- **K-2 PASS** (no task beyond 2σ regression).
- **K-3 FAIL** (weak-task average −0.10 < +1.5).
- **K-4 PASS**.

**Decision.** Park the 45/5/40/10 mixture. No seed-1502 replication, no ratio
tuning. The treatment moved BoolQ/WinoGrande, not OpenBookQA/ARC-C/LAMBADA/
HellaSwag. The public base remains v0.3.1.

Next: **X50**, a 50/50 weight soup of v0.3.1 and the X49 treatment, evaluated
on the same suite as a cheap chance to convert the near-miss. If it fails, the
cross-modal binding discovery is the next window.

## D94 — X50 soup: null (2026-11-08)

**Result.** 50/50 v0.3.1 + X49-treatment soup: mean **0.4284** vs v0.3.1
**0.4258** (+0.26 pp); weak-task average **−0.13 pp**. S-1/S-3 fail; S-2
passes; no task beyond 2σ regression.

**Decision.** No v0.3.2, no soup tuning. Public base remains v0.3.1. Next:
cross-modal binding discovery.

## D95 — KALIA-Vision v2 image captioning passes (2026-11-08)

**Result.** From-scratch CNN encoder + 4 continuous image tokens + frozen
v0.3.1 LoRA caption decoder on synthetic two-object scenes. 2,328,832
trainable params, 8 epochs, 24.4 minutes. Held-out exact **1.000** (1,000
images); blind 0.006; shuffled-image 0.002 (drop 0.998); all four attributes
1.000. V2-1…V2-5 pass.

**Decision.** Release **v0.5.2-vision-see** capability update. Public base
remains v0.3.1. Caveat: all 144 two-object combinations are in training; this
is seen-combination captioning. Next: hold out combinations and test
compositional generalization.

## D96 — KALIA-Vision v2.1 compositional split passes (2026-11-08)

**Result.** 29 of 144 two-object combinations held out (20%, seed 2026);
39,853 train, 4,015 seen val. Held-out exact **0.9920**; seen exact **0.9970**;
gap **0.0050**; held-out attributes 0.992–1.000; blind 0.000; shuffled 0.003.
2.33M trainable; 20.3 minutes. C-1…C-5 pass.

**Decision.** Release **v0.5.3-vision-compose** capability update. Public base
remains v0.3.1. This is evidence of compositional binding on the synthetic
grid, not open-world vision. Next: see→draw→see cycle consistency.

## D97 — Vision see→draw→see cycle: null (2026-11-08)

**Result.** 1,000 real images; captioner v2.1 → v1 diffusion → captioner v2.1.
Cycle exact **0.032**; attribute consistency 0.278–0.476. X-1/X-2 fail;
X-3/X-4 pass.

**Interpretation.** The v1 classifier recognizes generated images at 0.948, so
generation is not random; the captioner trained on real images does not read
the generated distribution. The gap is at the captioner input distribution.

**Decision.** No release. Next: cycle-consistency fine-tune of the captioner on
diffusion-generated images with known source captions, then re-test held-out
cycle. If it fails, the gap is a boundary at this scale.

## D98 — X53 cycle-consistency fine-tune: null (2026-11-08)

**Result.** 10k generated seen + 10k real replay, 2 epochs. Real held-out
caption exact **0.984**; held-out cycle exact **0.005**; cycle attributes
0.275–0.665. X-1/X-2 fail; X-3/X-4 pass.

**Interpretation.** The captioner still reads real held-out images, but not
generated held-out combinations. The see→draw gap is compositional, not a
generic distribution shift. Generated-seen adaptation does not transfer.

**Decision.** No release, no tuning. The next direction is joint training or a
captioner-consistency loss on the diffusion, not more replay.
