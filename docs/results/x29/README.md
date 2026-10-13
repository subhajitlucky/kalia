# X29 — patha replay: raw artifacts

Registered: `docs/preregistrations/2026-10-11-X29-patha-replay.md` (sha `8a1bae46…`, ledger-verified).
One continual update of v0.2.0 per arm, seed 1401, steps 4770 → 5770, `rewarm`,
1000 steps, replay fixed at 10% of every batch, identical except replay order.

| arm | state | val@5700 | kernel |
|---|---|---|---|
| uniform (control) | complete | 2.8076 | subhajitlucky/kalia-x29-uniform |
| jata | complete | 2.8510 | subhajitlucky/kalia-x29-jata |
| ghana | complete | 2.8752 | subhajitlucky/kalia-x29-ghana |
| krama | launched 2026-10-13 | — | subhajitlucky/kalia-x29-krama |

Artifacts per completed arm: `val_log.csv` (per 100 steps), `train_log.csv`
(per 10 steps). Final checkpoints live in the kernel outputs
(`out/x29-<arm>/ckpt.pt`) and are scored by `tools/kalia_forget_probe_kernel.py`
(`kalia-x29-forget-probe`), the registered CL-0 retention endpoint.

## val curve (val.bin, the same protocol as X26/X28)

| step | uniform | jata | ghana |
|---|---|---|---|
| 4800 | 2.7680 | 2.7766 | 2.7858 |
| 4900 | 2.6816 | 2.7025 | 2.7153 |
| 5000 | 2.9769 | 3.0137 | 3.0180 |
| 5100 | 2.7717 | 2.8129 | 2.8181 |
| 5200 | 2.7998 | 2.8543 | 2.8593 |
| 5300 | 2.8412 | 2.8953 | 2.8996 |
| 5400 | 2.6628 | 2.7181 | 2.7327 |
| 5500 | 2.8324 | 2.8869 | 2.9128 |
| 5600 | 2.8297 | 2.8788 | 2.8990 |
| 5700 | 2.8076 | 2.8510 | 2.8752 |

**The val ordering is monotone in structure depth, at every checkpoint from
5,000 on:** uniform (no structure) < jata (two-word braid) < ghana (nested
triples). Final gaps: jata−uniform **+0.0434**, ghana−jata **+0.0242**,
ghana−uniform **+0.0676** — all far above the ≤0.003 session-noise bound.
The deeper and denser the recitation structure of the replay stream, the
worse the held-out loss. Session noise bound (X28, same protocol): ≤ 0.003
per checkpoint. Context: the X28 no-replay static arm ended 2.8201 at the
same step.

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

| row | probe_loss | drift vs pretrain best |
|---|---|---|
| v0.2.0 baseline (step 4770) | 2.798875 | — |
| **uniform** (step 5770) | **2.775169** | **−0.023706** |
| **jata** (step 5770) | **2.816978** | **+0.018103** |

Baseline measured 2026-10-07 at step 4770; both arms measured 2026-10-12 at
step 5770. Drift is signed from the ledger's best-ever (the baseline):
negative = better than the pretrain's own score on the frozen probe.

**The arm gap is 0.0418 nats** — same instrument, same session, and 40,000×
the self-check's reproducibility. Retention and val **order the arms
identically** (uniform better, jata worse, by ~0.04 on both). No claim yet:
n=1 per arm, ghana/krama pending, and the pre-registration requires a second
seed for any winner.

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

