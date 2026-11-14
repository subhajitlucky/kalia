# X64 v150 full run

Status: **prepared, starts 2026-11-14.** Smoke passed: init val 2.7731,
throughput 6.5–6.9k tok/s on 2×T4. Public base remains v0.3.1.

## Session log

| session | date | steps | tokens | session val | tok/s | notes |
|---:|---|---:|---:|---:|---:|---|
| 1 | 2026-11-14 | running | | | ~7.1–7.6k | expand + init gate; started 07:28 UTC |
| 2 | 2026-11-15 | | | | | |
| 3 | 2026-11-16 | | | | | |
| 4 | 2026-11-17 | | | | | |
| 5 | 2026-11-21 | | | | | |
| 6 | 2026-11-22 | | | | | |
| 7 | 2026-11-23 | | | | | |
| 8 | 2026-11-24 | | | | | |
| 9 | 2026-11-28 | | | | | |
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

**Security:** the `kalia-hf-token` Kaggle dataset contains the HF write token.
Delete the dataset and rotate the token after the 120h run.

## Artifacts

- Config: `configs/kalia-v150-session.yaml`
- Kernel: `subhajitlucky/kalia-x64-v150-full`
- HF checkpoints: `kalia-lm/kalia-v150-run`
- Runbook: `docs/plans/2026-11-14-x64-full-run.md`
