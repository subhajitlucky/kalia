# X29 — patha replay: raw artifacts

Registered: `docs/preregistrations/2026-10-11-X29-patha-replay.md` (sha `8a1bae46…`, ledger-verified).
One continual update of v0.2.0 per arm, seed 1401, steps 4770 → 5770, `rewarm`,
1000 steps, replay fixed at 10% of every batch, identical except replay order.

| arm | state | val@5700 | retention drift | kernel |
|---|---|---|---|---|
| uniform (control) | complete | **2.8076** | **−0.0237** | subhajitlucky/kalia-x29-uniform |
| krama | complete | 2.8314 | −0.0007 | subhajitlucky/kalia-x29-krama |
| jata | complete | 2.8510 | +0.0181 | subhajitlucky/kalia-x29-jata |
| ghana | complete | 2.8752 | +0.0446 | subhajitlucky/kalia-x29-ghana |

All four registered arms ran (x29 uniform/jata/ghana/krama, one kernel each).

Artifacts per completed arm: `val_log.csv` (per 100 steps), `train_log.csv`
(per 10 steps). Final checkpoints live in the kernel outputs
(`out/x29-<arm>/ckpt.pt`) and are scored by `tools/kalia_forget_probe_kernel.py`
(`kalia-x29-forget-probe`), the registered CL-0 retention endpoint.

## val curve (val.bin, the same protocol as X26/X28)

| step | uniform | krama | jata | ghana |
|---|---|---|---|---|
| 4800 | 2.7680 | 2.7708 | 2.7766 | 2.7858 |
| 4900 | 2.6816 | 2.7018 | 2.7025 | 2.7153 |
| 5000 | 2.9769 | 3.0246 | 3.0137 | 3.0180 |
| 5100 | 2.7717 | 2.8199 | 2.8129 | 2.8181 |
| 5200 | 2.7998 | 2.8410 | 2.8543 | 2.8593 |
| 5300 | 2.8412 | 2.8788 | 2.8953 | 2.8996 |
| 5400 | 2.6628 | 2.6966 | 2.7181 | 2.7327 |
| 5500 | 2.8324 | 2.8663 | 2.8869 | 2.9128 |
| 5600 | 2.8297 | 2.8557 | 2.8788 | 2.8990 |
| 5700 | 2.8076 | 2.8314 | 2.8510 | 2.8752 |

**The val ordering is monotone in structure depth** — uniform (no structure,
adjacent windows) < krama (adjacent pairs, no reversal) < jata (two-word
braid) < ghana (nested triples) — at the final checkpoint and at **every**
checkpoint from 5,200 on. The exceptions are 5,000 and 5,100, where krama
spikes above jata: the same mid-run region where the step-5,000 val spike
appears in every arm. Final gaps: krama−uniform **+0.0238**, jata−krama
**+0.0196**, ghana−jata **+0.0242** — three near-equal ~0.02 steps, every one
far above the ≤0.003 session-noise bound, total span uniform→ghana **+0.0676**.
Context: the X28 no-replay static arm ended 2.8201, so shuffled replay helps
and structure depth is paid for rung by rung.

The step-5000 spike (~+0.2 on both arms) is reproducible — X28's static rerun
reproduced its own 5000 spike to 0.0007 — so it is a property of the protocol
at that step, not noise.

## Primary metric: CL-0 retention (registered)

Measured on the **G2-pinned probe shard** (`corpus/probe_val.bin`, sha256
`1664f71a…`, pinned in `docs/legal/val-shard-hashes.json`, fetched from
`kalia-lm/kalia-v020`), protocol identical to the 2026-10-07 baseline:
25 × 8 × 1024 tokens, seed 1234, CPU. Self-check: re-scoring the v0.2.0
checkpoint reproduced the recorded baseline to **0.000001** (2.798876 vs
2.798875). Ledger: `forgetting_x29.jsonl`.

| row | probe_loss | drift vs pretrain baseline |
|---|---|---|
| v0.2.0 baseline (step 4770) | 2.798875 | — |
| **uniform** (step 5770) | **2.775169** | **−0.023706** |
| **krama** (step 5770) | **2.798185** | **−0.000690** |
| **jata** (step 5770) | **2.816978** | **+0.018103** |
| **ghana** (step 5770) | **2.843516** | **+0.044641** |

Baseline measured 2026-10-07 at step 4770; the arms measured 2026-10-13 at
step 5770 (across two probe sessions that reproduced each other's rows to the
last decimal). The drifts above are all referenced to the **baseline** so the
rungs are comparable; the archived ledger's own `drift_from_best` column is
sequential (each row is scored against the best so far), so it differs for
rows scored after a better arm. probe_loss is the stable quantity.

**Retention is monotone in structure depth, on the same rungs as val:**
uniform −0.0237 < krama −0.0007 < jata +0.0181 < ghana +0.0446. Krama — the
mildest structure, no reversal — sits essentially at the baseline. The two
instruments agree not just in direction but in **rung sizes**:

| rung | Δ val | Δ retention |
|---|---|---|
| uniform→krama | +0.0238 | +0.0230 |
| krama→jata | +0.0196 | +0.0188 |
| jata→ghana | +0.0242 | +0.0265 |

Three near-equal ~0.02 steps on each instrument, from two different
evaluation sets and protocols. ghana's +0.0446 is the largest single CL-0
drift this project has measured. No claim yet: n=1 per arm, and the
pre-registration requires a second seed for any winner.

## Second seed (1402): the replication, and the no-replay arm scored for the first time

`tools/kalia_forget_probe_kernel.py` cohort `x29-replication`, kernel
`subhajitlucky/kalia-x29-replication-probe`, code `fc19ecd`. The cohort carries
the seed-1401 anchors as well as the seed-1402 pair, because the X28 static arm
had never been scored on CL-0 at either seed — the path heuristic in the old
probe could not name a checkpoint at `out/kautilya-static/`, so attaching it
made the baseline assertion count two and the run refused to start.

Self-check before any arm was believed: v0.2.0 re-scored **2.798875 vs recorded
2.798875, delta 0.000000**. The `uniform-s1401` anchor reproduced the archived
row exactly (2.775169, −0.023706). Probe sha `1664f71a…` asserted at run time.

| arm | val @5700 | CL-0 probe_loss | drift vs baseline |
|---|---|---|---|
| uniform-s1401 (10% replay) | 2.8076 | 2.775169 | −0.023706 |
| uniform-s1402 (10% replay) | 2.8058 | 2.776687 | −0.022188 |
| static-s1401 (**no replay**) | 2.8201 | **2.686720** | **−0.112155** |
| static-s1402 (**no replay**) | 2.8131 | **2.682753** | **−0.116122** |

Ledger: `replication/forgetting_x29_replication.jsonl`. Its `drift_from_best`
column is sequential against the best row so far and is **not** comparable
across arms; the drifts above are re-referenced to the baseline row.

Seed spreads: 0.0018 / 0.0070 on val (uniform / static) and 0.0015 / 0.0040 on
CL-0 — far below the 58M pretraining control spread of 0.2892 (D48).

**The two instruments reproduce across seeds and disagree with each other.**
val favours replay by +0.0125 (s1401) and +0.0073 (s1402), the second inside the
static arm's own seed spread. CL-0 favours no-replay by 0.0884 (s1401) and
0.0939 (s1402) — 22× the largest seed spread in the cohort, same direction at
both seeds.

Two confounds, both recorded before the result was read: the arms also differ in
delivery path and corpus passes (uniform ≈0.2 passes over 2.4B tokens, static
≈1.05 over 4 × 124,500,691 equalised shards at 60/20/15/5 with
`adaptive_every: 0`; both spend ~524M tokens), and `corpus/probe_val.bin` is a
frozen prefix of the training stream rather than a held-out set. **Not
concluded: that replay harms retention.** See journal 2026-10-18. The M0 gate —
"uniform 10% replay + rewarm beats no-replay beyond the spread", read on both
instruments — is not passed.

## The no-replay control: what that 0.09 actually was

`x29-noreplay` (kernels `kalia-x29-noreplay`, `-s1402`, code `6f06390`) ran the
X29 runner at `REPLAY = 0.00` — same runner, same 2.4B `train.bin`, same
1,000-step budget as `uniform`, so replay is the only variable. Probe cohort
`x29-noreplay` (kernel `kalia-x29-noreplay-probe`, code `64ee1cf`), ledger
`noreplay/forgetting_x29_noreplay.jsonl`. Self-check 2.798875 vs recorded
2.798875 at delta 0.000000; both uniform anchors reproduced exactly.

| arm | replay | val @5700 (s1401 / s1402) | CL-0 probe_loss (s1401 / s1402) |
|---|---|---|---|
| uniform | 10% | 2.8076 / 2.8058 | 2.775169 / 2.776687 |
| **noreplay** | **0%** | **2.8064 / 2.8078** | **2.769569 / 2.771140** |
| static | 0% (shards) | 2.8201 / 2.8131 | 2.686720 / 2.682753 |

The uniform→static gap decomposes almost entirely into something other than
replay:

| seed | uniform−static (CL-0) | replay's part | delivery path's part |
|---|---|---|---|
| 1401 | 0.088449 | **0.005600 (6.3%)** | 0.082849 (93.7%) |
| 1402 | 0.093934 | **0.005547 (5.9%)** | 0.088387 (94.1%) |

**Replay's own effect.** On CL-0 it is a small, reproducible *penalty*:
+0.005600 and +0.005547, same sign and near-identical size at both seeds, ~3.6×
the measured seed spread (0.0015). On val it is undetectable: +0.0012 and
−0.0020, sign-unstable and inside the 0.0018 spread.

**Withdrawn.** The earlier line in this file — "the X28 no-replay static arm
ended 2.8201, so shuffled replay helps" — stands on the confounded comparison
and is retracted. The gap is the corpus and the passes over it. The val
advantage of `uniform` over `static` (+0.0125 / +0.0073) is a delivery-path
effect, not a replay effect.

M0's gate is therefore not passed on either instrument.



## Incident record: the probe almost scored the wrong instrument (v1–v3)

The first three probe-kernel versions scored `data/probe/val_forget.bin` (the
2026-09-28 pretrain-era split, sha `7b71e339…`) instead of the G2-pinned
`corpus/probe_val.bin` (`1664f71a…`). Re-scoring v0.2.0 on the wrong file
gave 2.909614 — **+0.1107** against the recorded row — and the built-in
self-check refused to score any arm until the right instrument was used.
That +0.11 gap is the two files' difficulty difference; it also means a
silent substitution would have shipped a fake 0.11-nat "regression" on every
arm. The same producer/consumer probe-file mismatch has a recorded precedent
(`test_publish_contract.py`: "producer wrote val_forget.bin, consumer wanted
probe_val.bin"); this was its third sighting. Fix in `d220918`; the kernel
now fetches the pinned shard from the hub and refuses any other hash.

