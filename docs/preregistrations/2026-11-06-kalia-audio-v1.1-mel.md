# Pre-registration — KALIA-Audio v1.1: log-STFT 2D CNN

Registered: 2026-11-06, before the v1.1 GPU run.

## Context

Audio v1 (raw-waveform 1D CNN, 12 real Speech Commands words) reached
**0.655** test accuracy, below the 0.80 bar. The model is too weak for raw
speech. v1.1 changes the front end to a log-STFT 2D CNN, which is the standard
strong baseline for this dataset.

## Data

Same Google Speech Commands v0.02, 12 words:
`yes, no, up, down, left, right, on, off, stop, go, zero, one`.
Official `testing_list.txt`; train capped at 1,000/class (~10.8k);
test ~4.9k. No pretrained audio weights.

## Model

- Log-magnitude STFT: 16 kHz, n_fft 400, hop 160, 201×101 time-frequency map.
- 2D CNN: 32/64/128/256 conv blocks, global pool, 12-way head.
- ~0.5M parameters, trained from scratch, 30 epochs, AdamW.

## Bars

| ID | prediction | threshold |
|---|---|---|
| **R-1** | held-out real-speech accuracy | ≥ 0.80 |
| **R-2** | no pretrained audio weights | pass |
| **R-3** | all twelve commands present | pass |
| **R-4** | artifacts | checkpoint, metrics, per-class table |

## Decision rule

- **R-1…R-3 pass:** release **v0.6.1-audio-speech** capability update.
- **Fail:** log the boundary; no release. Do not tune after the result.
- Next: KALIA response generation, then TTS.

## Cost

≤ 1 GPU-h; fits the current window.
