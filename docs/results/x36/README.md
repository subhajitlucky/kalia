# X36 — the entity memory keeps names, but costs language quality

Registered: `docs/preregistrations/2026-10-27-X36-entity-memory-micro.md`
(sha `ea7cb66c…`, ledger-verified). Kernels
`subhajitlucky/kalia-x36-memory` and `subhajitlucky/kalia-x36-entity`, code
`523c577` + `9dfc6ae`, **~5.4 GPU-h** (4.6h training + 0.8h entity probe).

The model can generate the right continuation but cannot retain or select it.
X36 tested a different bet: a **self-supervised entity workspace** written from
the model's final states, read at the output, with a **copy head**, against a
**parameter-matched adapter control**.

## Design

30M micro, 6 layers × 384 dim × 6 heads, context 512, 763 steps, seeds
1338/1339. Four arms × two seeds:

| arm | what | params |
|---|---|---:|
| `base` | standard micro MLP | 29,921,280 |
| `control` | parameter-matched adapter, no memory | 30,707,712 |
| `copy` | copy head only | 30,216,577 |
| `memory` | 32-slot workspace + copy + recurrence | 30,697,378 |

Control vs memory differ by **0.03%**. The workspace writes at 64-token chunk
ends from the chunk state, the recurrence probability and the model's own NLL;
the read is added at the output with a zero-initialised projection. The
recurrence head is trained on the self-supervised "will this token appear again
in the next 64 tokens?" label, masked to token ids above 1000.

## Training result

| arm | val s1338 | val s1339 | mean |
|---|---:|---:|---:|
| base | 4.2393 | 4.2389 | 4.2391 |
| **control** | **3.7707** | **3.8840** | **3.8274** |
| copy | 3.9899 | 4.1831 | 4.0865 |
| **memory** | 3.9378 | 3.8688 | **3.9033** |

- **M-2 FAILS**: memory − control = **+0.0760 nats**, against a +0.010
  allowance. The control's extra capacity is a strong baseline; the memory
  spends its capacity on the workspace.
- Recurrence AUC: **0.823 / 0.824** (M-3 passes; the write signal is learnable).
- Copy accuracy on recurring in-prefix tokens: 2.4–3.5% (memory), 3.2–3.6%
  (copy). The copy head alone is weak.

## Entity result (short-context protocol)

20 stories, 384-token prompt, 128 generated, first half 64 / second half 64.
**Not comparable in absolute terms to the 58M X34 protocol**; the paired
control/memory contrast is the measurement.

| arm | greedy | half-pick | oracle |
|---|---:|---:|---:|
| base | 0.000 / 0.002 | 0.006 / 0.000 | 0.010 / 0.008 |
| control | 0.000 / 0.000 | 0.008 / 0.007 | 0.019 / 0.033 |
| copy | 0.008 / 0.013 | 0.066 / 0.041 | 0.109 / 0.116 |
| **memory** | **0.011 / 0.016** | **0.074 / 0.090** | **0.142 / 0.174** |

- **M-1 PASS**: memory greedy − control = **+0.0106 / +0.0162**, both seeds,
  bar 0.005 (control spread 0.0).
- **M-1b PASS**: memory oracle is far above control (0.142/0.174 vs
  0.019/0.033).
- **M-4 PASS**: distinct-2 is not degraded (memory 0.127 vs base 0.120; copy
  0.118).

The memory does exactly what it was built to do: it keeps names. The copy head
alone already lifts the candidate pool (oracle 0.109/0.116); the workspace
adds more (0.142/0.174). The greedy continuation stays low (1–2%), so the
selector problem is still there.

## Registered verdict

| ID | bar | result |
|---|---|---|
| **M-1** | memory greedy − control ≥ max(control spread, 0.005), both seeds | **PASS** |
| **M-1b** | oracle not degraded | **PASS** |
| **M-2** | memory val − control ≤ +0.010 | **FAIL (+0.076)** |
| **M-3** | recurrence AUC > 0.60 | **PASS (0.823)** |
| **M-4** | distinct-2 not worse by >20% | **PASS** |

**`advance = false`.** The pre-registered decision for "M-1 passes, M-2 fails"
is: **the memory buys retention at a language cost; report the trade-off, no
scale.** No 58M run is registered from this result.

## What this establishes

- **A self-supervised entity workspace can improve entity retention** at 30M:
  greedy +0.011/+0.016, half-pick 0.074/0.090, oracle 0.142/0.174, all against
  a parameter-matched control. The recurrence head learns the write signal
  (AUC 0.823).
- **It is not a free win.** The memory costs +0.076 nats against the control,
  far beyond the +0.010 allowance. The control's extra dense capacity is a
  strong baseline; the memory's capacity is spent on the workspace.
- **The copy head alone has a retention signal** (oracle 0.109/0.116), but its
  accuracy is weak and its val is worse; the workspace + recurrence adds the
  larger oracle/half-pick gain.

## What this does not establish

- Not a 58M result; the micro is a screen, and the short-context protocol is
  not the X34 protocol.
- Not a selector; the greedy continuation is still 1–2%.
- Not a scale-up: the pre-registered rule says no.
- Not evidence that the memory is useless; it is evidence that this version
  pays too much language quality for the retention gain.

## What would come next, if anything

A **val-preserving** memory is a new registration, not a tune of this one.
Candidates: a much smaller workspace, a memory-only-on-entity-tokens gate, an
auxiliary LM-preservation loss, or a copy head with a stronger calibration.
The X36 result says the mechanism is real; the cost is the open problem.

## Artifacts

- `x36_train.json` — training vals, recurrence AUC, copy accuracy.
- `x36_entity.json` / `x36_verdict.json` — per-story entity rows and the verdict.
- `run_manifest-train.json` / `run_manifest-entity.json` — provenance.
- `out/<arm>-s<seed>/val_log.csv` — the eight val curves.
- `configs/x36-<arm>-s<seed>.yaml` — the eight configs.
- `kalia-x36-memory.log`, `kalia-x36-entity.log` — the kernel logs.
