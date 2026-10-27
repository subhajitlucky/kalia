# X35 — the static code mixture does not move the 58M update

Registered: `docs/preregistrations/2026-10-26-X35-static-mixture-screen.md`
(sha `bbbb3414…`, ledger-verified). Kernel
`subhajitlucky/kalia-x35-mix-screen`, version 1, code `8cd8c79`, one GPU session
(**7.71 GPU-h**).

The only large positive in the record is data: the compliance-rebuilt corpus
moved held-out loss by 0.169 nats at matched tokens. D19 planned a **55/20/15/10**
mixture and never ran it. X26/X28 showed the *adaptive* policy drifts to easy
sources and costs +0.0183; X29's 94% “delivery path” effect was confounded with
corpus passes. X35 held the shards, schedule, token budget and seed fixed and
moved only the static source weights.

## Design

- v0.2.0 at step 4,770; 500 update steps; re-warm; no replay; canonical
  `kalia-prep-v2b` `val.bin`.
- The four published equalised per-source shards from `kalia-lm/kalia-v020`
  (124,500,691 tokens each, 498,002,764 total).
- Primary metric: the last logged val point, **step 5,200** with
  `eval_interval: 100` (Amendment 1; the final 5,270 is not an eval boundary).
- TinyStories (20%) and Cosmopedia (15%) fixed; FineWeb absorbs the change.

| arm | FineWeb | TinyStories | Cosmopedia | Python |
|---|---|---|---|---|
| `control` | 0.60 | 0.20 | 0.15 | 0.05 |
| `code10` | 0.55 | 0.20 | 0.15 | **0.10** |
| `code25` | 0.40 | 0.20 | 0.15 | **0.25** |

## Results

| arm | val @ 5,200 | val − control | train @ 5,200 | train − control |
|---|---|---|---|---|
| `control` | **2.7903** | — | 2.7511 | — |
| `code10` | 2.7896 | **−0.0007** | 2.6502 | −0.1009 |
| `code25` | 2.7961 | **+0.0058** | 2.2515 | **−0.4996** |

The val curve is flat across the mixtures at every checkpoint:

| step | control | code10 | code25 |
|---|---|---|---|
| 4,800 | 2.7989 | 2.7991 | 2.8031 |
| 4,900 | 2.6876 | 2.6895 | 2.6956 |
| 5,000 | 2.9801 | 2.9817 | 2.9902 |
| 5,100 | 2.7665 | 2.7677 | 2.7698 |
| 5,200 | 2.7903 | 2.7896 | 2.7961 |

The control's step-5,200 value is **0.0137 below** the archived X28 static
step-5,200 anchor (2.8040), inside the pre-registered 0.02 sanity bound.

## Registered bars

| ID | bar | measured | verdict |
|---|---|---|---|
| **S-1** | `code10 − control ≤ −0.010` at seed 1401 | **−0.0007** | **FAIL** |
| **S-2** | second seed, same bar | — | **not run** (contingent on S-1) |
| **S-3** | code dose-response; collapse if `code25 − control > +0.050` | **+0.0058** | within boundary |
| **S-4** | CL-0 loss not worse than control by > 0.0070 | — | **not run** (no Stage 2) |
| **S-5** | per-source held-out table | — | **not measured** (see below) |

**Registered verdict: S-1 fails. Do not run Stage 2. Close the static-mixture
track at this budget.** The D19 5-point shift does nothing; the 20-point shift
is slightly worse on val.

## The dessert signature

The code-heavy arm's **training** loss is **0.50 nats lower** than control at the
same step, while its held-out val is 0.0058 *worse*. The 10%-code arm is 0.10
nats lower on train and 0.0007 better on val. More code makes the training
mixture easier to fit; it does not make the held-out distribution better. This is
the X26/X28 dessert pattern at the aggregate level, not a data win.

**S-5 was not measured.** The static path (`adaptive_every: 0`) does not write
`per_source_log.csv`; that file is produced only on a reweight boundary. The
aggregate train/val divergence is reported instead. A future static screen must
add a standalone per-source eval at the end, or the gaming check is silently
unavailable.

## What this establishes

- **The static code share is not a cheap lever at 58M in a 500-step continual
  update.** The D19 target is flat; a 20-point increase is slightly worse.
- **A 5-point mixture change is below this screen's resolution**, and a 20-point
  change does not move val in the right direction.
- **The train/val dessert signature is visible in aggregate**: lower training
  loss on the reweighted mixture, no held-out gain. Any future mixture work must
  report both.

## What this does not establish

- Not a full-scale data test. 500 steps, not v0.2.0's 4,770; a 1,000-step or
  full-scale effect is not ruled out.
- Not the long-form test. PG-19/Gutenberg was probed for document lengths but no
  training shard exists yet; it targets LAMBADA and story structure, not code
  share.
- Not a claim that mixture never matters. It is a claim that this cheap static
  code screen did not move the canonical val, and the heavier code arm moved
  train loss only.
- Not a CL-0 result: Stage 2 did not run, so retention was not measured.

## Next

No Stage 2; the 22.29h remaining are not spent on a failed screen. The static
mixture track is closed at this budget. The next data bet is either:

1. **The long-form corpus** — the one data hypothesis aimed at the remaining
   benchmark failure (LAMBADA) and at story structure; it needs a CPU tokenised
   shard and a new 500-step registration, not another code-share arm.
2. **A full-scale v0.4 data run** — the only way to test the 0.169-nat corpus
   effect at scale; paired seeds, WSD, ~12.6 GPU-h.

Neither is launched here. The screen's result is the input to that decision, not
a substitute for it.

## Artifacts

- `x35_screen.json` — the registered verdict and the three arms.
- `run_manifest.json` — code fingerprint, shard/checkpoint/val hashes, weights.
- `out/<arm>-s1401/{val_log,train_log}.csv` — the curves.
- `configs/x35-<arm>-s1401.yaml` — the configs the kernel built.
- `kalia-x35-mix-screen.log` — the kernel log.

The verdict is reproduced from `x35_screen.json`; the decision rule is in the
pre-registration.
