"""Synthetic spoken commands for KALIA-Audio v0.

Six commands, 16 kHz, 0.6 s. Each command is a unique sequence of vowel-like
formant segments with a consonant-noise burst. No external audio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

SR = 16000
DUR = 0.6
N = int(SR * DUR)
COMMANDS = ("left", "right", "up", "down", "stop", "go")
# Vowel formant pairs (F1, F2) in Hz.
VOWELS = {
    "a": (800, 1200),
    "i": (300, 2300),
    "u": (350, 800),
    "e": (500, 1800),
    "o": (450, 900),
}
# Unique vowel sequence per command.
SEQUENCES = {
    "left": ("i", "e"),
    "right": ("a", "i"),
    "up": ("u", "o"),
    "down": ("a", "u"),
    "stop": ("e", "o"),
    "go": ("o", "u"),
}


def synth(command: str, rng: random.Random) -> np.ndarray:
    seq = SEQUENCES[command]
    f0 = rng.uniform(110, 190)
    speed = rng.uniform(0.85, 1.15)
    out = np.zeros(N, dtype=np.float32)
    # consonant burst
    burst_len = int(0.04 * SR)
    burst = np.random.default_rng(rng.randrange(1 << 30)).uniform(-1, 1, burst_len).astype(np.float32)
    burst *= np.hanning(burst_len).astype(np.float32)
    out[:burst_len] += 0.25 * burst
    pos = int(0.06 * SR)
    seg_len = int((N - pos) / (len(seq) + 0.5))
    for vowel in seq:
        f1, f2 = VOWELS[vowel]
        length = int(seg_len * speed * rng.uniform(0.9, 1.1))
        length = min(length, N - pos)
        if length <= 0:
            break
        t = np.arange(length, dtype=np.float32) / SR
        # gliding formants slightly over the segment
        glide = np.linspace(1.0, rng.uniform(0.9, 1.1), length).astype(np.float32)
        wave = (
            0.55 * np.sin(2 * np.pi * f0 * t)
            + 0.30 * np.sin(2 * np.pi * f1 * glide * t)
            + 0.15 * np.sin(2 * np.pi * f2 * glide * t)
        )
        env = np.hanning(length).astype(np.float32)
        out[pos:pos + length] += (0.6 * wave * env).astype(np.float32)
        pos += length
    out += np.random.default_rng(rng.randrange(1 << 30)).uniform(-0.01, 0.01, N).astype(np.float32)
    out += np.random.default_rng(rng.randrange(1 << 30)).normal(
        0, 0.005, N
    ).astype(np.float32)
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.9).astype(np.float32)


def generate(n: int, seed: int):
    rng = random.Random(seed)
    wav = np.empty((n, N), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)
    rows = []
    for i in range(n):
        idx = rng.randrange(len(COMMANDS))
        command = COMMANDS[idx]
        wav[i] = synth(command, rng)
        labels[i] = idx
        rows.append({"id": i, "command": command})
    return wav, labels, rows


def audit(wav: np.ndarray, labels: np.ndarray, rows: list[dict]) -> dict:
    counts = {c: int((labels == i).sum()) for i, c in enumerate(COMMANDS)}
    return {
        "n": len(wav),
        "shape": list(wav.shape),
        "sr": SR,
        "duration": DUR,
        "min": float(wav.min()),
        "max": float(wav.max()),
        "mean_abs": float(np.abs(wav).mean()),
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"splits": {}}
    for name, n, seed in (("train", args.n_train, 31), ("val", args.n_val, 32),
                          ("test", args.n_test, 33)):
        wav, labels, rows = generate(n, seed)
        np.save(args.out_dir / f"{name}_wav.npy", wav)
        np.save(args.out_dir / f"{name}_labels.npy", labels)
        with open(args.out_dir / f"{name}_labels.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        manifest["splits"][name] = audit(wav, labels, rows)
        manifest["splits"][name]["wav_sha256"] = hashlib.sha256(
            (args.out_dir / f"{name}_wav.npy").read_bytes()
        ).hexdigest()
        print(name, manifest["splits"][name]["n"], flush=True)
    (args.out_dir / "audit.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: v["n"] for k, v in manifest["splits"].items()}))


if __name__ == "__main__":
    main()
