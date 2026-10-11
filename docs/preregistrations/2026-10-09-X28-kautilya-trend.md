# X28 — Kautilya trend policy vs level policy vs static, and the X26 replication

**Status:** registered 2026-10-09, before launch (quota reset just occurred).
**Depends on:** X26 pilot (screen; artifacts in `docs/results/x26-pilot/`), the
trend implementation (`mixture.py`, config-wired), the pilot's DDP fixes.

## What this run is

Three arms, all resumed from v0.2.0 at step 4770, seed 1401, `rewarm: true`,
1000 update steps, identical in everything except the policy:

- **static** (`adaptive_every: 0`) — the control, and the **X26 static-arm
  rerun**: the run-to-run reproducibility check the pilot deferred.
- **level** (`adaptive_every: 100`, `trend_window: 1`) — the X26 policy
  exactly, as the replication arm.
- **trend** (`adaptive_every: 100`, `trend_window: 3`, `trend_eps: 0.01`,
  `collapse_margin: 0.5`) — the A1 rule: improving → dana, flat → sama,
  worsening → bheda, collapsed → danda.

## What is measured

- Primary: held-out val on the canonical `val.bin`, every 100 steps.
- **The run-to-run noise, directly:** static-rerun vs the X26 static arm
  (2.8200). Every delta in this run is read against 1x that number — this is
  the paired-noise bound the X26 pre-registration deferred to.
- The diary (`per_source_log.csv`) for level and trend: per-source losses,
  resulting weights, and the strategy applied per source.

## Registered expectations

- **Mechanism prediction:** the trend arm does **not** exhibit the monotonic
  dessert drift; its weights stay near 60/20/15/5, because a source being easy
  is no longer a reward.
- **Capability:** trend ≈ level ≈ static on val (null) is the expected
  outcome. The mechanism-level contrast (drift vs no-drift) is the primary
  read.
- If the trend arm shows the same drift, either the implementation is wrong or
  the signal is too noisy at ten decisions — report and fix; do not promote.

## Stopping and verdict

Session quota only. Any verdict on a capability delta requires the fresh-seed
(1402) replication of the winning arm; nothing is promoted from this run alone.

## Cost

~14 GPU-h (three arms x ~4.7h), plus CPU evaluation.
