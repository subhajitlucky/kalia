# Pre-registration — KALIA-Audio v1: real speech commands

Registered: 2026-11-06, before the real-speech GPU run.

## Goal

Replace the synthetic formant ear with a **real speech** command recognizer,
still trained from scratch. No pretrained audio model, no distillation.

## Data

Google Speech Commands v0.02 (public audio dataset; external data, not a
pretrained model). Twelve words:

`yes, no, up, down, left, right, on, off, stop, go, zero, one`

- Official `testing_list.txt` test split.
- Training: the remaining clips in the twelve command folders, capped at
  1,000 per class (~12k clips).
- 16 kHz, 1 s, raw waveform.
- No pretrained audio weights.

## Model

- 1D CNN on raw waveform (32/64/128/256 conv blocks, global pool, 12-way head).
- ~136k parameters, trained from scratch, 15 epochs, AdamW.

## Bars

| ID | prediction | threshold |
|---|---|---|
| **R-1** | held-out real-speech accuracy | ≥ 0.80 |
| **R-2** | no pretrained audio weights | pass |
| **R-3** | all twelve commands present | pass |
| **R-4** | artifacts | checkpoint, metrics, per-class table |

## Decision rule

- **R-1…R-3 pass:** release **v0.6.1-audio-speech** capability update.
- **Any fail:** log the boundary; no release. Do not tune after the result.
- Next: KALIA response generation from recognized commands, then TTS.

## Cost

Data download/extract CPU; training ≤ 1 GPU-h. Fits the current window.
