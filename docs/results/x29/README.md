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

