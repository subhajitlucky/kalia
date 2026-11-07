# KALIA-Audio v1.1 — real speech, log-STFT 2D CNN

Status: **clean pass of R-1…R-3.** Twelve real Speech Commands words, trained
from scratch. This replaces the v1 raw-waveform failure (0.655) with a proper
spectrogram front end.

## Result

- Held-out test accuracy: **0.8561** (4,891 clips).
- Validation: **0.8517** (1,200 clips).
- 391,884 parameters; 30 epochs in **5.7 minutes**.
- Checkpoint sha `e5fd5c33…`.
- Data: 10,800 train (1,000/class), official `testing_list.txt`.
- No pretrained audio weights.

| bar | result |
|---|---|
| **R-1** real-speech accuracy ≥ 0.80 | **PASS (0.856)** |
| **R-2** no pretrained audio | **PASS** |
| **R-3** all twelve commands | **PASS** |

## What changed from v1

v1 used a raw-waveform 1D CNN and reached only **0.655**. v1.1 uses a
log-magnitude STFT (201×101) and a 2D CNN; same data, same from-scratch rule.
The spectrogram front end is the difference.

## Caveat

Twelve words, 1,000 clips/class, a small 392k model. It is a real speech
command recognizer, not open-vocabulary speech understanding. The next step is
the 35-word Speech Commands vocabulary, then KALIA response generation and TTS.

## Artifacts

- `metrics.json` — test/val, params, hash.
- `train.log` — full training log.
- Kernel: `subhajitlucky/kalia-audio-v1-1`.
- Pre-registration: `docs/preregistrations/2026-11-06-kalia-audio-v1.1-mel.md`
  (hash `93b417eb…`).
