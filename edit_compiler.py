"""X41 — Edit-Compile-Then-Continue (ECC).

A LoRA edit compiler turns `instruction + factual prefix` into the
post-intervention prefix; the frozen v0.3.1 reader then scores candidate
continuations in that edited world.

The compiler is trained on single interventions plus non-test pair types. The
four benchmark pair types are held out. This module is deliberately separate
from the model: the reader is never trained and its checksum is checked.

Usage (inside the Kaggle GPU kernel):
    from edit_compiler import ...
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from intervention_pairs import (
    NAMES_EVAL,
    NAMES_TRAIN,
    TRAIN_TEMPLATES,
    InterventionDataset,
    _record,
    causal_pad_mask,
)
from eval_ip import accuracy_from_scores, score_candidates

PAD_ID = 50256
NEWLINE_ID = 198
EOS_ID = 50256

# Non-test pair types only. The benchmark holds out rename+swap,
# swap+negate, number+place and rename+negate.
ALLOWED_PAIRS = (
    ("rename", "place"),
    ("swap", "place"),
    ("place", "negate"),
)
TEST_PAIRS = (
    ("rename", "swap"),
    ("swap", "negate"),
    ("number", "place"),
    ("rename", "negate"),
)

# Extra templates add pair coverage without ever using a test pair type.
# count_place is singles-only (number+place is a test pair).
EXTRA_TEMPLATES = [
    {
        "id": "carry_place",
        "prefix": "{A} carried the {adj} {noun} {prep} the {surface}.",
        "suffix": "{B} saw the {noun} {prep} the {surface}.",
        "neg_prefix": "{A} did not carry the {adj} {noun}.",
        "neg_suffix": "{B} did not see the {noun}.",
        "ops": ("rename", "swap", "place", "negate"),
        "has_c": False,
        "place": ("on", "under"),
    },
    {
        "id": "give_place",
        "prefix": "{A} gave {B} the {adj} {noun} {prep} the {surface}.",
        "suffix": "{B} kept the {noun} {prep} the {surface}.",
        "neg_prefix": "{A} did not give {B} the {adj} {noun}.",
        "neg_suffix": "{A} kept the {noun}.",
        "ops": ("rename", "swap", "place", "negate"),
        "has_c": False,
        "place": ("on", "under"),
    },
    {
        "id": "count_place",
        "prefix": "{A} placed {num} {noun_plural} {prep} the {surface}.",
        "suffix": "There are {num} {noun_plural} {prep} the {surface}.",
        "ops": ("number", "place"),
        "has_c": False,
        "number": ("two", "three"),
        "place": ("on", "under"),
    },
]

TARGETS = {"qkv", "proj", "gate", "up", "down"}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _pairs_for(template: dict, allowed: tuple[tuple[str, str], ...]):
    return [p for p in allowed if all(op in template["ops"] for op in p)]


def _dedup(records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in records:
        key = (r["instruction"], r["prefix"], r["edited_prefix"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def build_records(
    templates: list[dict],
    names: list[str],
    *,
    singles_per_op: int,
    pairs_per_type: int,
    allowed_pairs: tuple[tuple[str, str], ...] = (),
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    for template in templates:
        for op in template["ops"]:
            for i in range(singles_per_op):
                try:
                    out.append(_record(rng, names, template, [op], "train", i))
                except ValueError:
                    continue
        for pair in _pairs_for(template, allowed_pairs):
            for i in range(pairs_per_type):
                try:
                    out.append(_record(rng, names, template, list(pair), "train", i))
                except ValueError:
                    continue
    return _dedup(out)


def build_train(arm: str, singles_per_op: int = 1000, pairs_per_type: int = 1000,
                seed: int = 41) -> list[dict]:
    templates = list(TRAIN_TEMPLATES) + list(EXTRA_TEMPLATES)
    if arm == "B":
        return build_records(
            templates, NAMES_TRAIN,
            singles_per_op=singles_per_op,
            pairs_per_type=pairs_per_type,
            allowed_pairs=ALLOWED_PAIRS,
            seed=seed,
        )
    if arm == "A":
        return build_records(
            templates, NAMES_TRAIN,
            singles_per_op=singles_per_op,
            pairs_per_type=0,
            allowed_pairs=(),
            seed=seed,
        )
    raise ValueError(f"unknown arm {arm!r}")


def build_dev(arm: str, singles_per_op: int = 120, pairs_per_type: int = 120,
              seed: int = 999) -> list[dict]:
    templates = list(TRAIN_TEMPLATES)
    if arm == "B":
        return build_records(
            templates, NAMES_EVAL,
            singles_per_op=singles_per_op,
            pairs_per_type=pairs_per_type,
            allowed_pairs=ALLOWED_PAIRS,
            seed=seed,
        )
    return build_records(
        templates, NAMES_EVAL,
        singles_per_op=singles_per_op,
        pairs_per_type=0,
        allowed_pairs=(),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# format
# ---------------------------------------------------------------------------
def compiler_prompt(record: dict) -> str:
    return f"{record['instruction']}\nFactual: {record['prefix']}\nEdited:"


def compiler_target(record: dict) -> str:
    return f" {record['edited_prefix']}\n"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.A(x)) * self.scale


def apply_lora(model: nn.Module, rank: int = 16, alpha: int = 16) -> int:
    """Wrap target Linear layers in place; freeze every base parameter."""
    for p in model.parameters():
        p.requires_grad = False
    targets = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(module, nn.Linear) and leaf in TARGETS:
            targets.append((name, module))
    for name, module in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, LoRALinear(module, rank, alpha))
    trainable = [p for p in model.parameters() if p.requires_grad]
    return sum(p.numel() for p in trainable)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if k.endswith(".A.weight") or k.endswith(".B.weight")
    }


def load_lora(model: nn.Module, path: str | Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("lora", payload)
    load = model.load_state_dict(state, strict=False)
    assert not load.unexpected_keys, load.unexpected_keys
    assert not any(k.endswith((".A.weight", ".B.weight")) for k in load.missing_keys), (
        load.missing_keys
    )


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
class CompilerDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], enc, max_len: int = 160):
        self.items: list[tuple[list[int], list[int], list[bool]]] = []
        for r in records:
            p = enc.encode_ordinary(compiler_prompt(r))
            t = enc.encode_ordinary(compiler_target(r))
            full = p + t
            if len(full) > max_len:
                continue
            input_ids = full[:-1]
            labels = full[1:]
            labels[: len(p) - 1] = [-100] * (len(p) - 1)
            loss_mask = [False] * (len(p) - 1) + [True] * len(t)
            self.items.append((input_ids, labels, loss_mask))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


def collate(items, max_len: int = 160):
    b = len(items)
    x = torch.full((b, max_len), PAD_ID, dtype=torch.long)
    y = torch.full((b, max_len), -100, dtype=torch.long)
    mask = torch.zeros(b, max_len, dtype=torch.bool)
    token_mask = torch.zeros(b, max_len, dtype=torch.bool)
    for i, (ids, labels, lm) in enumerate(items):
        n = len(ids)
        x[i, :n] = torch.tensor(ids)
        y[i, :n] = torch.tensor(labels)
        mask[i, :n] = torch.tensor(lm)
        token_mask[i, :n] = True
    return x, y, mask, token_mask, causal_pad_mask(token_mask)


# ---------------------------------------------------------------------------
# generation / evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_edited_prefix(model, record: dict, enc, device, max_new: int = 64) -> str:
    was_training = model.training
    model.eval()
    ids = enc.encode_ordinary(compiler_prompt(record))
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out: list[int] = []
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.cfg.context_len:])
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        tok = int(nxt.item())
        if tok in (NEWLINE_ID, EOS_ID):
            break
        out.append(tok)
        idx = torch.cat([idx, nxt], dim=1)
    text = enc.decode(out).strip()
    if was_training:
        model.train()
    return normalize(text)


def exact_match(records: list[dict], generated: list[str]) -> dict:
    n = len(records)
    correct = sum(
        normalize(g) == normalize(r["edited_prefix"])
        for r, g in zip(records, generated)
    )
    nonempty = sum(bool(g.strip()) for g in generated)
    changed = sum(
        normalize(g) != normalize(r["prefix"])
        for r, g in zip(records, generated)
    )
    return {
        "n": n,
        "exact_match": correct / n if n else float("nan"),
        "nonempty": nonempty / n if n else float("nan"),
        "changed_from_factual": changed / n if n else float("nan"),
    }


def generate_all(model, records: list[dict], enc, device, max_new: int = 64) -> list[str]:
    return [generate_edited_prefix(model, r, enc, device, max_new) for r in records]


def score_bench(base, bench_path: Path, out_path: Path, rows_path: Path,
                prompt_mode: str, device, batch_size: int = 16) -> dict:
    ds = InterventionDataset(bench_path, prompt_mode=prompt_mode)
    scores = score_candidates(base, ds, device, batch_size=batch_size)
    acc = accuracy_from_scores(ds, scores)
    payload = {
        "prompt_mode": prompt_mode,
        "n": len(ds.records),
        "by_cell": acc["by_cell"],
        "by_operation": acc["by_operation"],
        "seen": acc["seen"],
        "unseen": acc["unseen"],
        "overall": acc["overall"],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with open(rows_path, "w") as fh:
        for row in acc["rows"]:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return payload


def score_generated(base, records: list[dict], generated: list[str],
                    tmp_path: Path, out_path: Path, rows_path: Path,
                    device, batch_size: int = 16) -> dict:
    with open(tmp_path, "w") as fh:
        for r, g in zip(records, generated):
            r2 = dict(r)
            r2["edited_prefix"] = g if g.strip() else r["prefix"]
            fh.write(json.dumps(r2, sort_keys=True) + "\n")
    payload = score_bench(base, tmp_path, out_path, rows_path,
                          "edited_prefix", device, batch_size)
    payload["compiler"] = exact_match(records, generated)
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _lr(step: int, total: int, base_lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (min_ratio + (1 - min_ratio) * coeff)


def train_compiler(
    model,
    records: list[dict],
    dev_records: list[dict],
    enc,
    out_dir: Path,
    *,
    seed: int,
    steps: int,
    batch_size: int = 32,
    lr: float = 2e-4,
    warmup: int = 100,
    eval_every: int = 300,
    eval_n: int = 150,
    device,
    label: str = "armB",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    ds = CompilerDataset(records, enc)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=collate, num_workers=0,
    )
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_em = -1.0
    best_step = -1
    best_state = None
    train_log = []
    dev_log = []
    step = 0
    t0 = time.time()
    model.train()
    while step < steps:
        for x, y, loss_mask, token_mask, attn_mask in loader:
            if step >= steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            lr_now = _lr(step, steps, lr, warmup)
            for group in opt.param_groups:
                group["lr"] = lr_now
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                _, loss = model(x, targets=y, loss_mask=loss_mask,
                                attn_mask=attn_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(opt)
            scaler.update()
            if step % 20 == 0:
                train_log.append({"step": step, "loss": float(loss.item()),
                                  "lr": lr_now, "seconds": time.time() - t0})
                print(f"{label} seed {seed} step {step}/{steps} "
                      f"loss {loss.item():.4f} lr {lr_now:.2e}", flush=True)
            if step % eval_every == 0 or step == steps - 1:
                model.eval()
                sample = dev_records[:eval_n]
                gen = generate_all(model, sample, enc, device, max_new=64)
                em = exact_match(sample, gen)["exact_match"]
                dev_log.append({"step": step, "dev_em": em, "seconds": time.time() - t0})
                print(f"{label} seed {seed} step {step} dev_em {em:.4f}", flush=True)
                if em > best_em:
                    best_em = em
                    best_step = step
                    best_state = lora_state_dict(model)
                model.train()
            step += 1
    if best_state is None:
        best_state = lora_state_dict(model)
    payload = {
        "lora": best_state,
        "seed": seed,
        "label": label,
        "steps": steps,
        "best_step": best_step,
        "best_dev_em": best_em,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_examples": len(ds),
        "dev_examples": len(dev_records),
    }
    path = out_dir / f"lora_{label}_seed{seed}.pt"
    torch.save(payload, path)
    with open(out_dir / f"train_{label}_seed{seed}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "loss", "lr", "seconds"])
        w.writeheader()
        w.writerows(train_log)
    with open(out_dir / f"dev_{label}_seed{seed}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "dev_em", "seconds"])
        w.writeheader()
        w.writerows(dev_log)
    print(f"{label} seed {seed} best dev_em {best_em:.4f} at step {best_step}; "
          f"adapter {path} ({path.stat().st_size} bytes)", flush=True)
    return {"path": str(path), "best_dev_em": best_em, "best_step": best_step,
            "trainable_params": payload["trainable_params"],
            "train_examples": len(ds), "dev_examples": len(dev_records)}
