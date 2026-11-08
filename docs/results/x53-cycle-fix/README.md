# X53 — cycle-consistency fine-tune: null

Status: **null.** Generated-image adaptation on seen combinations does not
transfer to held-out generated combinations.

## Result

- 10,000 generated training images from **seen** captions + 10,000 real replay;
  2 epochs, 24.3 minutes.
- Real held-out caption exact after fine-tune: **0.984**.
- Held-out cycle exact: **0.005**.
- Cycle attributes: left shape 0.665, left color 0.509, right shape 0.387,
  right color 0.275.
- X-1/X-2 fail; X-3/X-4 pass.
- Checkpoint sha `18ec3342…`.

## Interpretation

The captioner still reads **real** held-out images almost perfectly (0.984),
so the fine-tune did not destroy real-image ability. But it cannot read the
**generated held-out combinations**: the adaptation learned generated *seen*
combinations, not the compositional mapping. The see→draw gap is
**compositional**, not a generic distribution shift.

## Decision

No release. No tuning. A different direction is needed: joint training or a
captioner-consistency loss on the diffusion, not more generated-image replay.

## Artifacts

- `metrics.json`, `train.log`.
- Pre-registration: `docs/preregistrations/2026-11-08-X53-cycle-fix.md`
  (hash `a2793bed…`).
- Kernel: `subhajitlucky/kalia-x53-cycle-fix`.
