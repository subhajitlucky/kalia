# v0.6.1-audio-speech — KALIA-Audio v1.1

**Capability update — 2026-11-07.** KALIA's audio module now recognizes twelve
**real** spoken words, trained from scratch. No pretrained audio model.

## Result

- Held-out test accuracy: **0.8561** (4,891 clips).
- Validation: **0.8517**.
- 391,884 parameters; 30 epochs in **5.7 minutes**.
- Twelve words: yes, no, up, down, left, right, on, off, stop, go, zero, one.
- Official Speech Commands test split.
- Checkpoint sha `e5fd5c33…`.

| bar | result |
|---|---|
| R-1 accuracy ≥ 0.80 | **PASS (0.856)** |
| R-2 no pretrained audio | **PASS** |
| R-3 all twelve commands | **PASS** |

## What changed

v1 raw-waveform 1D CNN reached only 0.655. v1.1 uses a log-magnitude STFT
(201×101) + 2D CNN; same data, same from-scratch rule. The front end is the
difference.

## Caveat

Twelve words, 1,000 clips/class, 392k parameters. Real command recognition, not
open-vocabulary speech understanding. Next: 35-word vocabulary, KALIA response
generation, then TTS.

## Record

- Result: `docs/results/kalia-audio-v1.1/README.md`
- Pre-registration: `docs/preregistrations/2026-11-06-kalia-audio-v1.1-mel.md`
- Licence: repository `LICENSE`.
