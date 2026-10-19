# X17 — matched-eval rerun: the fix is undetectable, and the earlier "helps" was premature

Re-run of `kalia-x17-docmask` (kernel version 3, code `e10564f`) after
`evaluate()` was fixed to apply the attention mask a masked run trained under.
The four arms, two seeds, 500 steps, one kernel. Everything else identical to
the registered run.

## Why it was re-run

The first run measured the treatment **+0.0444 nats worse**, but both arms were
evaluated with full causal attention: the control was trained and tested matched,
and the treatment was tested in a regime it had never trained in. A 2×2 on the
archived checkpoints then put the matched delta at **−0.0124** and the mismatch
at −0.0570 — a reversal, but measured on *final losses only* with a different
protocol (100 fixed windows against the registered 20).

That left the registration's own second bar untested: **DM-2**, "the effect is
visible early, not just at the end". A single late crossing is as likely to be
noise as a mechanism, and only a curve can tell.

## The curves

| step | ctl s1338 | ctl s1339 | trt s1338 | trt s1339 | Δ s1338 | Δ s1339 |
|---|---|---|---|---|---|---|
| 100 | 6.0072 | 5.8677 | 6.0535 | 5.8362 | +0.0463 | −0.0315 |
| 200 | 5.3172 | 5.2847 | 5.3240 | 5.2646 | +0.0068 | −0.0201 |
| 300 | 5.0578 | 5.0719 | 5.0649 | 5.0558 | +0.0071 | −0.0161 |
| 400 | 4.8456 | 4.6203 | 4.8491 | 4.6064 | +0.0035 | −0.0139 |
| 500 | 4.8810 | 4.9082 | 4.8841 | 4.8949 | +0.0031 | −0.0133 |

## The registered bars

| ID | bar | measured | verdict |
|---|---|---|---|
| **DM-1** | mean paired Δ ≤ −0.010 | **−0.0051** | **FAIL** |
| **DM-1b** | both deltas negative | **+0.0031 / −0.0133** | **FAIL — the seeds disagree in sign** |
| **DM-2** | ahead at ≥3 of 5 points, both seeds | 0/5 and 5/5 | **FAIL** |
| **DM-3** | step time ≤ 1.25× | ×1.048 | PASS |
| **DM-4** | mask engagement < 1.0 | 0.8061 | PASS |
| **DM-5** | paired spread (reported) | **0.0164** | — |

**REGISTERED VERDICT: do not promote.**

## What this establishes

**The +0.0444 was an artifact.** That survives: the first run tested a masked
model unmasked, and the 2×2 isolated the mismatch at −0.0570, larger than the
effect it was hiding.

**The benefit is not detectable at this scale.** The two seeds disagree in sign,
and the spread between them (0.0164) is three times the mean effect (0.0051).
Under D48 that is not a small effect — it is an effect below the ruler.

**The fix is free and correct.** DM-3 passes at ×1.048 and DM-4 confirms the mask
engages on 80.6% of causally-eligible attention. It simply does not measurably
help, so there is no case for changing the recipe.

## A correction to this record

The 2×2 entry above said the matched mean "would clear the −0.010 bar at
−0.0124". **That was premature and is retracted here.** It was measured on final
losses only, on a non-registered protocol, at 0.0024 above a bar; this run puts
the same contrast at −0.0051 with a sign disagreement between seeds. An effect
that does not survive a rerun on the registered protocol was never established,
and calling it a pass was the same over-reading this project spent the day
removing from its other claims.

The 2×2's actual contribution stands and is narrower than claimed: it showed the
first number was a mismatch artifact. It did not show that masking helps.

## Three answers, one lesson

| run | reported | what it was |
|---|---|---|
| first | **+0.0444** — masking hurts | a train/eval mismatch |
| 2×2 | **−0.0124** — masking helps | a small effect that did not replicate |
| matched rerun | **−0.0051** — cannot tell | the honest answer |

Each successive run had a better instrument than the one before it, and the last
one is the only one that could have detected DM-2's failure at all. The lesson is
not "the third number is true" — it is that **the first two were reported without
the instrument that could have contradicted them**, and only the third had one.

## Artifacts

`docs/results/x17-matched/` — four val curves, four train curves, the run
manifest, the kernel log. Reproduce the verdict with
`python tools/x17_verdict.py --dir docs/results/x17-matched`.
