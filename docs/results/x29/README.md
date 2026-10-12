# X29 — patha replay: raw artifacts

Registered: `docs/preregistrations/2026-10-11-X29-patha-replay.md` (sha `8a1bae46…`, ledger-verified).
One continual update of v0.2.0 per arm, seed 1401, steps 4770 → 5770, `rewarm`,
1000 steps, replay fixed at 10% of every batch, identical except replay order.

| arm | state | val@5700 | kernel |
|---|---|---|---|
| uniform (control) | complete | 2.8076 | subhajitlucky/kalia-x29-uniform |
| jata | complete | 2.8510 | subhajitlucky/kalia-x29-jata |
| ghana | launched 2026-10-12 | — | subhajitlucky/kalia-x29-ghana |
| krama | staged, next quota window | — | — |

Artifacts per completed arm: `val_log.csv` (per 100 steps), `train_log.csv`
(per 10 steps). Final checkpoints live in the kernel outputs
(`out/x29-<arm>/ckpt.pt`) and are scored by `tools/kalia_forget_probe_kernel.py`
(`kalia-x29-forget-probe`), the registered CL-0 retention endpoint.

## val curve (val.bin, the same protocol as X26/X28)

| step | uniform | jata |
|---|---|---|
| 4800 | 2.7680 | 2.7766 |
| 4900 | 2.6816 | 2.7025 |
| 5000 | 2.9769 | 3.0137 |
| 5100 | 2.7717 | 2.8129 |
| 5200 | 2.7998 | 2.8543 |
| 5300 | 2.8412 | 2.8953 |
| 5400 | 2.6628 | 2.7181 |
| 5500 | 2.8324 | 2.8869 |
| 5600 | 2.8297 | 2.8788 |
| 5700 | 2.8076 | 2.8510 |

jata − uniform: +0.009 (4800), **+0.041 … +0.055 at every checkpoint from
5100 on**. Session noise bound (X28, same protocol): ≤ 0.003 per checkpoint.
Context: the X28 no-replay static arm ended 2.8201 at the same step.

The step-5000 spike (~+0.2 on both arms) is reproducible — X28's static rerun
reproduced its own 5000 spike to 0.0007 — so it is a property of the protocol
at that step, not noise.

## Status

Secondary metric (val): recorded above, **no claim** — n=1 per arm, primary
metric pending, second seed required before any winner.
Primary metric (CL-0 retention): probe run in flight; ledger to be archived
here as `forgetting_x29.jsonl`.
