# X64 v150 full run

Status: **prepared, starts 2026-11-14.** Smoke passed: init val 2.7731,
throughput 6.5–6.9k tok/s on 2×T4. Public base remains v0.3.1.

## Session log

| session | date | steps | tokens | session val | tok/s | notes |
|---:|---|---:|---:|---:|---:|---|
| 1 | 2026-11-14 | 458 | 240,123,904 | 2.7629 | ~8.9k | init 2.7731 → 2.7629; 7.5h; checkpoint uploaded |
| 2 | 2026-11-15 | 899 | 471,334,912 | 2.7583 | ~8.7k | resumed from HF; +441 steps; checkpoint uploaded |
| 3 | 2026-11-15 | 1,340 | 702,545,920 | 2.7578 | ~8.5k | resumed; +441 steps; step-1000 val 2.7553 |
| 4 | 2026-11-15 | 1,661 | 870,842,368 | 2.7676 | ~8.5k | data replay bug; val regressed; reverted to session 3 |
| 5 | 2026-11-21 | 1,788 | 937,426,944 | 2.7185 | ~8.8k | resumed session 3; fixed sampler; −0.039 nats |
| 6 | 2026-11-22 | 2,221 | 1,164,443,648 | 2.6982 | ~8.4k | crossed S-1 (≤2.70); resumed session 5 |
| 7 | 2026-11-22 | 2,671 | 1,400,373,248 | 2.6789 | ~8.4k | S-1 margin 0.021; resumed session 6 |
| 8 | 2026-11-24 | 3,057 | 1,602,748,416 | 2.6619 | ~8.5k | 6.5h cap; S-1 margin 0.038 |
| 8b | 2026-11-24 | 3,079 | 1,614,282,752 | 2.6595 | ~8.5k | 25-min top-up; +22 steps |
| 9 | 2026-11-28 | 3,508 | 1,839,202,304 | 2.6450 | ~8.5k | S-1 margin 0.055; resumed session 8b |
| 10 | 2026-11-29 | | | | | |
| 11 | 2026-11-30 | | | | | |
| 12 | 2026-12-01 | | | | | |
| 13 | 2026-12-05 | | | | | |
| 14 | 2026-12-06 | | | | | |
| 15 | 2026-12-07 | | | | | eval / retry |
| 16 | 2026-12-08 | | | | | eval / retry |

## Final gates

| gate | threshold | result |
|---|---|---|
| S-1 held-out val | ≤ 2.70 | |
| S-2 9-task mean | ≥ 0.468 | |
| S-3 capability regression | all bars | |
| S-4 provenance | own weights, no distillation | |

## Execution notes (2026-11-14)

- **HF Secrets unavailable.** `UserSecretsClient` returned a connection error.
  Fallback: a private Kaggle dataset `subhajitlucky/kalia-hf-token` containing
  `token.txt`. The kernel reads it only if the secret is unavailable.
- **First full attempt failed** because the kernel passed
  `[python, train.py, ...]` to `torch.distributed.run`; torchrun tried to
  execute the Python ELF as a script. Fixed by passing `train.py` as the
  entrypoint.
- **Second full attempt failed** with a 600-second NCCL timeout at step 24:
  rank 0 was uploading a multi-GB checkpoint to HF while rank 1 waited at the
  next collective. Fixed by training without `--hub-repo` and uploading the
  final checkpoint from the kernel **after torchrun exits**.
- Session 1 started at **07:28 UTC**; init val **2.7731**; training running.

**Data-order bug (session 4).** The data generator was seeded with only
`(seed, rank)`. Every resumed session therefore restarted the same random
stream and replayed the same windows. Session 4's val regression
(2.7578 → 2.7676) is a data-repetition artifact, not a capacity result.
`train.py` now offsets the generator seed by the absolute `start_step`.
Session 5 resumes from the **session-3 checkpoint** (`f7a681d`) so the
replayed session-4 weights are not used. The session-4 checkpoint remains in
HF history but is not the resume point.

**Security:** the `kalia-hf-token` Kaggle dataset contains the HF write token.
Delete the dataset and rotate the token after the 120h run.

## Artifacts

- Config: `configs/kalia-v150-session.yaml`
- Kernel: `subhajitlucky/kalia-x64-v150-full`
- HF checkpoints: `kalia-lm/kalia-v150-run`
- Runbook: `docs/plans/2026-11-14-x64-full-run.md`
