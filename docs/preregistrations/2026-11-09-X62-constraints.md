# Pre-registration — X62: constraint-satisfying visual generation

Registered: 2026-11-09, before the X62 GPU run.

## Why not X61 first

X60’s selected images are read back by the captioner at **0.994**, while the
independent classifier judges them at **0.953**. The 4.1 pp gap is
**verifier–judge disagreement**, not a captioner-detectable failure. A
self-repair loop using the same captioner can only fix the 0.6% it can see; it
cannot fix the 4.1 pp it cannot. X61 is deferred until there is an independent
second verifier. The next real capability is X62.

## Idea

Give KALIA a **compositional constraint** and let it search its own visual
imagination for a scene that satisfies it. It samples candidate captions,
draws them, reads each one back with the frozen captioner, and keeps the first
readback that satisfies the constraint. No external reward model, no human
labels, no rule-based filter in generation; the constraint is a predicate on
the model’s own readback.

This is the see→draw→judge→choose loop applied to **constraint satisfaction**,
not just image quality.

## Design

- 64×64 three-object stack: X59 8.7M UNetLarge, X58 captioner, X59 6-head
  classifier as the independent judge.
- **12 constraints**, 12 trials each, **64 candidates** per trial, sampled
  uniformly from the 13,824 combinations.
- Success = any candidate’s captioner readback satisfies the constraint.
- Classifier agreement = the selected candidate’s classifier attributes also
  satisfy the constraint.
- Oracle = any candidate’s *source* caption satisfies the constraint (search
  space contains a solution).
- No training.

## Constraints

Exactly one red; no blue; all shapes the same; all colors the same; exactly
two red; left and right same color; center is a star; at least two circles;
all colors different; left is red or blue; center and right shapes differ;
exactly two shapes the same.

## Bars

| ID | prediction | threshold |
|---|---|---|
| **CS-1** | mean constraint success over 12 constraints | ≥ 0.80 |
| **CS-2** | classifier agreement on satisfied scenes | ≥ 0.90 |
| **CS-3** | no external reward; from scratch | pass |
| **CS-4** | gallery artifacts | pass |

## Decision rule

- **CS-1…CS-3 pass:** release **v0.11.0-constraints** with the gallery.
- **Any fail:** log the boundary; no tuning after the result.
- Public base remains v0.3.1.

## Cost

≤ 1.5 GPU-h.
