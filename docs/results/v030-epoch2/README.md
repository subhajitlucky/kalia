# v0.3.0 second epoch — training complete; final evaluation in progress

Registered: `docs/preregistrations/2026-10-27-v030-second-epoch.md`
(sha `7375a595…`, ledger-verified). Kernel `kalia-v030-epoch2`, code `b103d5d`,
**~10 GPU-h**.

The 2,000-step re-warm cosine continued v0.2.0 on the same 2.4B corpus, no
replay, seed 1401. The canonical 100-batch eval printed:

```text
val loss 2.7743 | bpB 0.9070 | tokens 819,200
```

- **E-1 PASS** — 2.7743 ≤ 2.8048, and **0.0505 nats** better than v0.2.0's
  2.8248.
- **E-2 / E-3 pending.** The first final-eval kernel died on a markdown parse
  bug after printing the val; the fixed CPU kernel (`kalia-v030-final-eval`) is
  rerunning the 2,000-sample benchmark and the CL-0 probe. The final verdict
  will be appended here.

## Training artifacts

- `out/val_log.csv` — the 100-batch in-training curve; the last point 2.8208 at
  6,700, the canonical 100-batch final 2.7743.
- `out/train_log.csv` — the 2,000 update steps.
- `run_manifest.json` — code revision, data/checkpoint hashes, schedule.
- `v030-epoch2.yaml` — the built config.
- `kalia-v030-epoch2.log` — the kernel log.
- The final eval output will be added when E-2/E-3 land.
