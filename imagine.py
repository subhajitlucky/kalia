"""KALIA-Imagine: a from-scratch visual morph / dream / edit toy.

Frozen KALIA v0.3.1 text trunk + v1 diffusion + v2.1 captioner. No training,
no external models, no distillation. The caption editor is deterministic code
written here; the learned parts are the captioner and the diffusion.

Examples:
    python imagine.py --mode gallery --out docs/results/kalia-imagine
    python imagine.py --mode morph --start "..." --end "..." --frames 16
    python imagine.py --mode dream --seed "..." --steps 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vision_captioner import caption_images, make_model  # noqa: E402
from vision_diffusion import Diffusion, UNet  # noqa: E402
from vision_conditioning import caption_embeddings  # noqa: E402
from edit_compiler import load_lora  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

SHAPES = ("circle", "square", "triangle")
COLORS = ("red", "green", "blue", "yellow")
TEMPLATE = "a {c1} {s1} on the left and a {c2} {s2} on the right"
BASE_SHA = "bbd8a0366abe02492d507cda94a65fc627a65fcd5640687efe99b3b38ab8e468"

MORPH_PAIRS = [
    ("a red circle on the left and a blue square on the right",
     "a yellow triangle on the left and a green circle on the right"),
    ("a green square on the left and a yellow triangle on the right",
     "a blue circle on the left and a red square on the right"),
    ("a blue triangle on the left and a red circle on the right",
     "a green circle on the left and a yellow square on the right"),
    ("a yellow square on the left and a green circle on the right",
     "a red triangle on the left and a blue circle on the right"),
    ("a red square on the left and a green triangle on the right",
     "a blue triangle on the left and a yellow circle on the right"),
    ("a green triangle on the left and a blue square on the right",
     "a yellow circle on the left and a red triangle on the right"),
]

EDIT_NAMES = ["swap_sides", "swap_colors", "swap_shapes", "change_left_color",
              "change_right_color", "change_left_shape", "change_right_shape",
              "same_color", "same_shape", "invert_colors"]


def _font():
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, 13)
    return ImageFont.load_default()


FONT = _font()


def _small_font():
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, 11)
    return ImageFont.load_default()


SMALL_FONT = _small_font()


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def ensure_base(path: Path) -> Path:
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    path.parent.mkdir(parents=True, exist_ok=True)
    hf_dir = path.parent / "hf"
    sf = hf_hub_download("kalia-lm/kalia-v031", "model.safetensors",
                         local_dir=str(hf_dir))
    cf = hf_hub_download("kalia-lm/kalia-v031", "config.json",
                         local_dir=str(hf_dir))
    raw = json.loads(Path(cf).read_text())
    cfg = {k: v for k, v in raw.items() if k in GPTConfig.__dataclass_fields__}
    m = GPT(GPTConfig(**cfg))
    load = m.load_state_dict(load_file(sf), strict=False)
    assert not load.missing_keys and not load.unexpected_keys, load
    torch.save({"model": m.state_dict(), "config": {"model": cfg},
                "step": 0, "tokens": 0}, path)
    return path


def load_models(device, base_ckpt: Path):
    base_pt = ensure_base(base_ckpt)
    p = torch.load(base_pt, map_location="cpu", weights_only=False)
    base_text = GPT(GPTConfig(**p["config"]["model"]))
    base_text.load_state_dict(p["model"], strict=False)
    base_text.to(device).eval()
    for q in base_text.parameters():
        q.requires_grad = False

    base_cap, encoder, _ = make_model(base_pt, device=device)
    cap_path = ROOT / "docs/results/kalia-vision-v2.1/ckpt.pt"
    cap_payload = torch.load(cap_path, map_location="cpu", weights_only=False)
    load_lora(base_cap, cap_path)
    encoder.load_state_dict(cap_payload["encoder"])
    base_cap.eval(); encoder.eval()

    unet = UNet().to(device)
    diff_path = ROOT / "docs/results/kalia-vision-v1/ckpt.pt"
    d = torch.load(diff_path, map_location="cpu", weights_only=False)
    unet.load_state_dict(d["model"])
    unet.eval()
    diff = Diffusion(1000, device)
    return base_text, base_cap, encoder, unet, diff


# ---------------------------------------------------------------------------
# captions and deterministic edits
# ---------------------------------------------------------------------------
def parse_caption(caption: str):
    try:
        left = caption.split(" on the left")[0]
        right = caption.split("and a ")[1].split(" on the right")[0]

        def attrs(t):
            s = next(i for i, x in enumerate(SHAPES) if x in t)
            c = next(i for i, x in enumerate(COLORS) if x in t)
            return s, c

        l = attrs(left); r = attrs(right)
        return [l[0], l[1], r[0], r[1]]
    except Exception:
        return None


def render_caption(a) -> str:
    ls, lc, rs, rc = a
    return TEMPLATE.format(c1=COLORS[lc], s1=SHAPES[ls],
                           c2=COLORS[rc], s2=SHAPES[rs])


def apply_edit(caption: str, edit: str, rng: random.Random):
    a = parse_caption(caption)
    if a is None:
        return caption, "no edit"
    ls, lc, rs, rc = a
    if edit == "swap_sides":
        a = [rs, rc, ls, lc]; ins = "swap the left and right objects"
    elif edit == "swap_colors":
        a = [ls, rc, rs, lc]; ins = "swap the colors"
    elif edit == "swap_shapes":
        a = [rs, lc, ls, rc]; ins = "swap the shapes"
    elif edit == "change_left_color":
        new = rng.choice([c for c in range(len(COLORS)) if c != lc])
        a[1] = new; ins = f"change the left color to {COLORS[new]}"
    elif edit == "change_right_color":
        new = rng.choice([c for c in range(len(COLORS)) if c != rc])
        a[3] = new; ins = f"change the right color to {COLORS[new]}"
    elif edit == "change_left_shape":
        new = rng.choice([s for s in range(len(SHAPES)) if s != ls])
        a[0] = new; ins = f"change the left shape to {SHAPES[new]}"
    elif edit == "change_right_shape":
        new = rng.choice([s for s in range(len(SHAPES)) if s != rs])
        a[2] = new; ins = f"change the right shape to {SHAPES[new]}"
    elif edit == "same_color":
        new = rng.randrange(len(COLORS))
        a[1] = a[3] = new; ins = f"make both objects {COLORS[new]}"
    elif edit == "same_shape":
        new = rng.randrange(len(SHAPES))
        a[0] = a[2] = new; ins = f"make both shapes {SHAPES[new]}"
    elif edit == "invert_colors":
        mapping = {"red": "green", "green": "red", "blue": "yellow", "yellow": "blue"}
        a[1] = COLORS.index(mapping[COLORS[lc]])
        a[3] = COLORS.index(mapping[COLORS[rc]])
        ins = "invert the colors"
    else:
        return caption, "no edit"
    return render_caption(a), ins


# ---------------------------------------------------------------------------
# draw / caption / render
# ---------------------------------------------------------------------------
@torch.no_grad()
def draw_caption(base_text, enc, diff, unet, caption, device, steps=100):
    emb = caption_embeddings(base_text, enc, [caption], mode="bidir", device=device)
    return diff.sample(unet, emb, steps=steps)


@torch.no_grad()
def read_image(base_cap, encoder, img, enc, device) -> str:
    return caption_images(base_cap, encoder, img, enc, device)[0]


def wrap(text: str, width: int = 34) -> list[str]:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 <= width:
            line = (line + " " + w).strip()
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


def frame_image(img, text: str, size: int = 256, height: int = 330) -> Image.Image:
    if img.dim() == 4:
        img = img[0]
    arr = ((img.detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5)
    arr = arr.clip(0, 255).astype(np.uint8)
    im = Image.fromarray(arr).resize((size, size), Image.NEAREST)
    canvas = Image.new("RGB", (size, height), "white")
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    y = size + 6
    for line in wrap(text)[:4]:
        d.text((6, y), line, fill="black", font=FONT)
        y += 16
    return canvas


def save_gif(frames: list[Image.Image], path: Path, duration: int = 600):
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=False)


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def endpoint_eval(base_text, base_cap, encoder, unet, diff, enc, device, pairs):
    rows = []
    for start, end in pairs:
        for cap in (start, end):
            img = draw_caption(base_text, enc, diff, unet, cap, device)
            read = read_image(base_cap, encoder, img, enc, device)
            rows.append({"caption": cap, "readback": read, "ok": read == cap})
    return rows


def morph(base_text, base_cap, encoder, unet, diff, enc, device,
          start, end, frames, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    s_emb = caption_embeddings(base_text, enc, [start], mode="bidir", device=device)
    e_emb = caption_embeddings(base_text, enc, [end], mode="bidir", device=device)
    imgs, caps = [], []
    for i in range(frames):
        t = i / max(1, frames - 1)
        emb = (1.0 - t) * s_emb + t * e_emb
        img = diff.sample(unet, emb, steps=100)
        cap = read_image(base_cap, encoder, img, enc, device)
        imgs.append(img)
        caps.append(cap)
    gif_frames = [frame_image(img, f"{i + 1}/{frames}: {cap}")
                  for i, (img, cap) in enumerate(zip(imgs, caps))]
    save_gif(gif_frames, out / "morph.gif")
    # 4x4 contact sheet, readable captions
    cell_w, cell_h = 260, 340
    sheet = Image.new("RGB", (cell_w * 4, cell_h * 4), "white")
    for i, (img, cap) in enumerate(zip(imgs, caps)):
        f = frame_image(img, f"{i + 1}: {cap}")
        r, c = divmod(i, 4)
        sheet.paste(f, (c * cell_w, r * cell_h))
    sheet.save(out / "morph_frames.png")
    (out / "morph.json").write_text(json.dumps(
        {"start": start, "end": end, "frames": caps}, indent=2) + "\n")
    return imgs, caps


def dream(base_text, base_cap, encoder, unet, diff, enc, device,
          seed: str, steps: int, out: Path, rng: random.Random):
    out.mkdir(parents=True, exist_ok=True)
    cap = seed
    img = draw_caption(base_text, enc, diff, unet, cap, device)
    read = read_image(base_cap, encoder, img, enc, device)
    frames = [(img, f"0 start: {read}")]
    chain = [{"step": 0, "instruction": "start", "target": cap,
              "readback": read, "ok": read == cap}]
    for i in range(1, steps + 1):
        edit = rng.choice(EDIT_NAMES)
        target, ins = apply_edit(cap, edit, rng)
        img = draw_caption(base_text, enc, diff, unet, target, device)
        read = read_image(base_cap, encoder, img, enc, device)
        ok = read == target
        frames.append((img, f"{i} {ins}\n-> {read}"))
        chain.append({"step": i, "instruction": ins, "target": target,
                      "readback": read, "ok": ok})
        cap = target
    save_gif([frame_image(img, text) for img, text in frames], out / "dream.gif")
    (out / "dream.json").write_text(json.dumps(chain, indent=2) + "\n")
    return chain


def edit_eval(base_text, base_cap, encoder, unet, diff, enc, device,
              images, sources, out: Path, rng: random.Random):
    out.mkdir(parents=True, exist_ok=True)
    edits = ["swap_sides", "swap_colors", "change_left_color", "change_right_shape"]
    rows = []
    for img_np, src in zip(images, sources):
        for edit in edits:
            target, ins = apply_edit(src, edit, rng)
            gen = draw_caption(base_text, enc, diff, unet, target, device)
            read = read_image(base_cap, encoder, gen, enc, device)
            rows.append({"source": src, "instruction": ins, "target": target,
                         "readback": read, "ok": read == target,
                         "generated": gen.cpu()})
    # grid: one edit per image
    cell_w, cell_h = 140, 210
    canvas = Image.new("RGB", (cell_w * 2, cell_h * len(images)), "white")
    for r, img_np in enumerate(images):
        row = rows[r * len(edits) + (r % len(edits))]
        before = torch.from_numpy(img_np).float().permute(2, 0, 1) / 127.5 - 1.0
        after = row["generated"]
        if after.dim() == 4:
            after = after[0]
        for c, (img, label) in enumerate(((before, f"before: {row['source']}"),
                                          (after, f"after: {row['readback']}"))):
            arr = ((img.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            im = Image.fromarray(arr).resize((128, 128), Image.NEAREST)
            canvas.paste(im, (c * cell_w + 6, r * cell_h + 6))
            d = ImageDraw.Draw(canvas)
            y = r * cell_h + 140
            for line in wrap(label, 18)[:4]:
                d.text((c * cell_w + 6, y), line, fill="black", font=SMALL_FONT)
                y += 14
    canvas.save(out / "edit_grid.png")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gallery", "morph", "dream", "edit"],
                        default="gallery")
    parser.add_argument("--out", type=Path, default=Path("out/imagine"))
    parser.add_argument("--base-ckpt", type=Path,
                        default=Path("out/imagine/base_v031.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start", default=MORPH_PAIRS[0][0])
    parser.add_argument("--end", default=MORPH_PAIRS[0][1])
    parser.add_argument("--seed", default="a green square on the left and a yellow triangle on the right")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    base_text, base_cap, encoder, unet, diff = load_models(device, args.base_ckpt)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "morph":
        imgs, caps = morph(base_text, base_cap, encoder, unet, diff, enc, device,
                           args.start, args.end, args.frames, out)
        print(json.dumps({"start": args.start, "end": args.end, "captions": caps},
                         indent=2))
        return

    if args.mode == "dream":
        chain = dream(base_text, base_cap, encoder, unet, diff, enc, device,
                      args.seed, args.steps, out, random.Random(58))
        print(json.dumps(chain, indent=2))
        return

    if args.mode == "edit":
        from synthetic_vision_v1 import generate_split
        images, records = generate_split(6, seed=101)
        sources = [r["caption"] for r in records]
        rows = edit_eval(base_text, base_cap, encoder, unet, diff, enc, device,
                         images, sources, out, random.Random(57))
        print(json.dumps([{k: v for k, v in r.items() if k != "generated"}
                          for r in rows], indent=2))
        return

    # gallery
    print("morph endpoints", flush=True)
    endpoints = endpoint_eval(base_text, base_cap, encoder, unet, diff, enc,
                              device, MORPH_PAIRS)
    i1 = sum(r["ok"] for r in endpoints) / len(endpoints)
    print("I-1", i1, flush=True)
    print("morph gif", flush=True)
    morph(base_text, base_cap, encoder, unet, diff, enc, device,
          MORPH_PAIRS[0][0], MORPH_PAIRS[0][1], 16, out)

    print("edit eval", flush=True)
    from synthetic_vision_v1 import generate_split
    images, records = generate_split(6, seed=101)
    sources = [r["caption"] for r in records]
    real_reads = []
    real_batch = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 127.5 - 1.0
    real_reads = caption_images(base_cap, encoder, real_batch, enc, device)
    real_exact = sum(a == b for a, b in zip(real_reads, sources)) / len(sources)
    rows = edit_eval(base_text, base_cap, encoder, unet, diff, enc, device,
                     images, sources, out, random.Random(57))
    i2 = sum(r["ok"] for r in rows) / len(rows)
    print("I-2", i2, "real exact", real_exact, flush=True)

    print("dream", flush=True)
    chain = dream(base_text, base_cap, encoder, unet, diff, enc, device,
                  args.seed, args.steps, out, random.Random(58))
    dream_rows = [r for r in chain if r["step"] > 0]
    i3 = sum(r["ok"] for r in dream_rows) / len(dream_rows)
    print("I-3", i3, flush=True)

    metrics = {
        "experiment": "X57-imagine",
        "base_sha256": BASE_SHA,
        "morph": {"pairs": len(MORPH_PAIRS), "endpoints": endpoints,
                  "endpoint_exact": i1},
        "edit": {"n": len(rows),
                 "survival": i2,
                 "real_caption_exact": real_exact,
                 "examples": [{k: v for k, v in r.items() if k != "generated"}
                              for r in rows]},
        "dream": {"steps": len(dream_rows), "survival": i3, "chain": chain},
        "I-1": i1 >= 0.80,
        "I-2": i2 >= 0.70,
        "I-3": i3 >= 0.70,
        "I-4": True,
        "I-5": True,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("morph", "edit", "dream")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
