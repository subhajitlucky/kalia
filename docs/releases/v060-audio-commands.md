# v0.6.0-audio-commands — KALIA-Audio v0

**Capability release — 2026-11-06.** KALIA's first audio input module: a
136k-parameter 1D CNN recognizes six synthetic spoken commands from raw 16 kHz
waveforms. Trained from scratch; no pretrained audio weights.

This is a capability module, not a new base model. KALIA v0.3.1 is unchanged.

## Result

- Held-out test accuracy: **0.912** (1,000 clips).
- Validation: **0.916**.
- 136,326 parameters; 15 epochs in **0.38 minutes**.
- Checkpoint sha `a6ea7051…`, 545KB.
- All six commands present; class 5 is the weakest at 0.766.

| bar | result |
|---|---|
| A-1 command accuracy ≥ 0.90 | **PASS (0.912)** |
| A-2 no pretrained audio weights | **PASS** |
| A-3 all six commands present | **PASS** |
| A-4 artifacts | **PASS** |

## What is released

- `ckpt.pt` — the 136k-parameter audio CNN.
- `synthetic_audio.py` — the formant command generator.
- `tools/kalia_audio_*_kernel.py` — data, training and per-class eval kernels.
- `docs/results/kalia-audio-v0/` — metrics, checkpoint, logs, pre-registration.

## Caveat

Toy vocabulary on **synthetic formant tones**, not real speech. It is the first
owned “ear”, not a speech recognizer. Next: real speech commands (external
audio data, still no pretrained model), then KALIA response generation and TTS.

## Record

- Result: `docs/results/kalia-audio-v0/README.md`
- Pre-registration: `docs/preregistrations/2026-11-06-kalia-audio-v0.md`
- Licence: repository `LICENSE`.
