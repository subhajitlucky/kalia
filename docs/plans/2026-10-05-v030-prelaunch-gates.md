# v0.3.0 pre-launch gates — everything that must be true before Step 0 launches

Date: 2026-10-05. Status: **checklist, not all gates pass.**
Companion: `2026-09-29-v030-continual-update.md` (the plan) and
`2026-10-05-v030-amendment-1.md` (the tightening). A gate is a binary
condition with a named checker. No gate is waived by judgment on launch day;
a gate that cannot pass is amended in writing first, per the project's own
process (X19-am1, X20-am1).

## G0 — Quota and artifacts exist (operational)

| check | pass criterion |
|---|---|
| Kaggle GPU quota available | ≥ 3.0h confirmed in the UI before launching anything (§6 of the amendment budgets ≈ 2.9h including the third Step-1 arm) |
| `corpus/probe_val.bin` published | present at `kalia-lm/kalia-v020`, byte-identical to the prep output (Status update 4; enforced by `test_publish_contract.py`) |
| `corpus/<source>.bin` replay shards published | same condition, all four sources |
| Step 1 kernel pairs all three control arms | one launch failure cannot silently halve the spread sample |

*Why:* the sequence costs one kernel per step; launching without the
artifacts repeats Status-update-4 (code-complete, unrunnable).

## G1 — Local suite green and ledger verified (CPU, free)

| check | pass criterion |
|---|---|
| `pytest` full suite | 292/292 passing (last verified 2026-10-05: 113 in `tests/` + 179 root-level) |
| `python tools/verify_prereg_ledger.py` | all entries verify, including `v030-am1` |

*Why:* the suite is the only thing standing between us and a repeat of the
re-warm-schedule defect class (three attempts, two worse-than-original
intermediates — all caught by tests, Status update 3).

## G2 — The yardstick is pinned (D43 must not recur)

D43 cost a false-regression scare because a corpus rebuild silently
redefined the val set. The v0.3.0 sequence adds new shards (long-form,
replay), which is exactly the operation that caused D43.

| check | pass criterion |
|---|---|
| sha256 of the canonical val shard recorded | hash written into `docs/legal/` or the v0.3.0 run log **before** Step 1, from the artifact Step 1 actually reads — not from the prep output |
| CL-0 probe shard hash recorded alongside it | same procedure for `corpus/probe_val.bin` |
| Any future corpus operation re-checks both hashes | follow-up work: a `test_val_shard_hash.py` asserting the recorded hashes, so the next rebuild fails loudly instead of D43-quietly |

*Why:* 11 minutes of CPU here versus a false-regression narrative that
would redirect the whole cycle. The test does not exist yet; the hashes
cannot be recorded until the artifacts exist (G0). G2 passes when both
hashes are on file.

## G3 — New shards pass the licence gate (I16 must not recur)

| check | pass criterion |
|---|---|
| long-form corpus licence status recorded | Gutenberg is public-domain; the sentence-aware chunking output is a derived artifact of the same tokens — record this reasoning in `docs/legal/training-data-licence-matrix.md`, do not assume it |
| replay shard licence status recorded | same filtered sources as v0.2.0, but the shard is a new artifact — the matrix gets a row for it |
| filter-gates-build wiring | the check of record: no corpus artifact used by Steps 1–4 may postdate its licence check (the I16 defect was a 40-minute ordering gap, not a missing filter) |

*Why:* v0.2.0's card discloses ~2% copyleft tokens because a filter landed
after the corpus it should have gated. v0.3.0 must not add a second
disclosure.

## G4 — Document masking stays OFF for v0.3.0, X17 stays queued

The intra-document masking fix exists (`doc_mask` flag, `test_doc_mask.py`,
`micro-docmask.yaml`) and X17 is registered — but X17 has never run, and
v0.2.0 trained with masking **off**.

| check | pass criterion |
|---|---|
| all seven v0.3.0 arm configs have `doc_mask: false` | verified by the diff-lock tests (`test_v030_configs.py`), same mechanism as the existing allow-list |
| X17 recorded as the v0.4.0 decision input, not a v0.3.0 variable | turning masking on mid-sequence would change the training distribution for all arms including the "static" control, voiding the control's meaning |

*Why:* the static arm is defined as "the v0.2.0 mixture" — same pipeline,
same masking. A pipeline change is a confound, not an improvement, inside
a continual-update measurement. Measure it at micro scale when quota allows;
decide for v0.4.0.

## G5 — Step 0 gates the sequence (Amendment 1, §4)

| check | pass criterion |
|---|---|
| CL-0 executed against the real v0.2.0 checkpoint on Kaggle | forgetting measurable on held-out val |
| probe-shows-nothing response | sequence stops, probe is fixed first — the numbers after a broken probe are not data |

## Gate status at writing

| gate | status |
|---|---|
| G0 | **open** — quota unverified, artifact presence unverified from here |
| G1 | **pass** — suite 292/292, ledger to be re-verified after this doc commits |
| G2 | **open** — hashes cannot be recorded until G0's artifacts exist |
| G3 | **open** — matrix rows for the new shards not yet written |
| G4 | **pass** — neither v0.2.0 nor any arm sets `doc_mask` (train.py defaults it False), and `test_v030_configs.py` fails any arm that adds it as drift outside the allow-list |
| G5 | **open** — IS Step 0; passes by running |

Launch order: G0 → G1 → G2/G3 (either order) → G4 verify → G5 runs first.
Amendment 1's §6 checklist is subsumed by G0.
