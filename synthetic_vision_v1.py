"""Synthetic two-object shapes dataset for KALIA-Vision v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle")
COLORS = {
    "red": (220, 40, 40),
    "green": (40, 180, 60),
    "blue": (40, 80, 220),
    "yellow": (240, 200, 40),
}
POSITIONS = {"left": 7, "right": 25}
TEMPLATE = "a {c1} {s1} on the left and a {c2} {s2} on the right"
SS = 4
SIZE = 32


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, color, cx: int, cy: int, r: int):
    if shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    elif shape == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=color)
    else:
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)


def _draw(rng: random.Random) -> tuple[np.ndarray, str]:
    s1, s2 = rng.choice(SHAPES), rng.choice(SHAPES)
    c1, c2 = rng.choice(list(COLORS)), rng.choice(list(COLORS))
    img = Image.new("RGB", (SIZE * SS, SIZE * SS), (238, 238, 238))
    draw = ImageDraw.Draw(img)
    for shape, color, pos in ((s1, c1, "left"), (s2, c2, "right")):
        cx = POSITIONS[pos] * SS + rng.randint(-1 * SS, 1 * SS)
        cy = 16 * SS + rng.randint(-1 * SS, 1 * SS)
        r = rng.randint(4, 5) * SS
        _draw_shape(draw, shape, COLORS[color], cx, cy, r)
    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    noise = np.random.default_rng(rng.randrange(1 << 30)).integers(
        -4, 5, size=arr.shape, dtype=np.int16
    )
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    caption = TEMPLATE.format(c1=c1, s1=s1, c2=c2, s2=s2)
    return arr, caption


def generate_split(n: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    rng = random.Random(seed)
    images = np.empty((n, SIZE, SIZE, 3), dtype=np.uint8)
    records = []
    for i in range(n):
        arr, caption = _draw(rng)
        images[i] = arr
        records.append({"id": i, "caption": caption})
    return images, records


def audit(images: np.ndarray, records: list[dict]) -> dict:
    counts = {}
    for r in records:
        counts[r["caption"]] = counts.get(r["caption"], 0) + 1
    return {
        "n": len(records),
        "shape": list(images.shape),
        "unique_captions": len(counts),
        "min_caption_count": min(counts.values()),
        "max_caption_count": max(counts.values()),
        "pixel_min": int(images.min()),
        "pixel_max": int(images.max()),
        "pixel_mean": float(images.mean()),
        "caption_counts": dict(sorted(counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-val", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=5000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"splits": {}}
    for name, n, seed in (("train", args.n_train, 21), ("val", args.n_val, 22),
                          ("test", args.n_test, 23)):
        images, records = generate_split(n, seed)
        np.save(args.out_dir / f"{name}_images.npy", images)
        with open(args.out_dir / f"{name}_captions.jsonl", "w") as fh:
            for r in records:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        manifest["splits"][name] = audit(images, records)
        manifest["splits"][name]["images_sha256"] = hashlib.sha256(
            (args.out_dir / f"{name}_images.npy").read_bytes()
        ).hexdigest()
        print(name, manifest["splits"][name]["n"], flush=True)
    (args.out_dir / "audit.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: v["n"] for k, v in manifest["splits"].items()}))


if __name__ == "__main__":
    main()
