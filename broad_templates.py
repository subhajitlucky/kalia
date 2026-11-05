"""X46 broad-template training data.

Seventeen templates: the seven X43 templates plus ten new broad templates.
Every 1-, 2- and 3-op combination supported by a template is generated. The
four X44 test templates are absent by construction.

The target is the post-intervention prefix; candidate sets are not needed for
training.
"""

from __future__ import annotations

import itertools
import random

from intervention_pairs import (
    NAMES_EVAL,
    NAMES_TRAIN,
    TRAIN_TEMPLATES,
    _instruction,
    _op_vector,
    _slots,
    render,
)
from edit_compiler import EXTRA_TEMPLATES

BROAD_TEMPLATES = [
    {
        "id": "mail_send",
        "prefix": "{A} mailed a {adj} {noun} to {B}.",
        "suffix": "{B} showed the {noun} to {C}.",
        "neg_prefix": "{A} did not mail a {adj} {noun} to {B}.",
        "neg_suffix": "{B} did not show the {noun} to {C}.",
        "ops": ("rename", "swap", "negate"),
        "has_c": True,
    },
    {
        "id": "borrow_back",
        "prefix": "{A} borrowed the {adj} {noun} from {B}.",
        "suffix": "{B} asked {A} for the {noun}.",
        "neg_prefix": "{A} did not borrow the {adj} {noun} from {B}.",
        "neg_suffix": "{B} did not ask {A} for the {noun}.",
        "ops": ("rename", "swap", "negate"),
        "has_c": False,
    },
    {
        "id": "plant_count",
        "prefix": "{A} planted {num} {noun_plural} {prep} the {surface}.",
        "suffix": "There are {num} {noun_plural} {prep} the {surface}.",
        "ops": ("number", "place"),
        "has_c": False,
        "number": ("two", "three"),
        "place": ("on", "under"),
    },
    {
        "id": "paint_hang",
        "prefix": "{A} painted a {adj} {noun} for {B}. {B} hung it {prep} the {surface}.",
        "suffix": "{A} saw the {noun} {prep} the {surface}.",
        "neg_prefix": "{A} did not paint a {adj} {noun} for {B}. {B} hung it {prep} the {surface}.",
        "neg_suffix": "{A} did not see the {noun}.",
        "ops": ("rename", "swap", "place", "negate"),
        "has_c": False,
        "place": ("on", "under"),
    },
    {
        "id": "cook_serve",
        "prefix": "{A} cooked {num} {noun_plural} for {B}.",
        "suffix": "{B} served {num} {noun_plural} {prep} the {surface}.",
        "ops": ("rename", "number", "place"),
        "has_c": False,
        "number": ("two", "three"),
        "place": ("on", "under"),
    },
    {
        "id": "carry_deliver",
        "prefix": "{A} carried the {adj} {noun} to {B}.",
        "suffix": "{B} delivered it to {C}.",
        "neg_prefix": "{A} did not carry the {adj} {noun} to {B}.",
        "neg_suffix": "{B} did not deliver it to {C}.",
        "ops": ("rename", "swap", "negate"),
        "has_c": True,
    },
    {
        "id": "write_read",
        "prefix": "{A} wrote a {adj} {noun} for {B}.",
        "suffix": "{B} read the {noun} to {C}.",
        "neg_prefix": "{A} did not write a {adj} {noun} for {B}.",
        "neg_suffix": "{B} did not read the {noun} to {C}.",
        "ops": ("rename", "swap", "negate"),
        "has_c": True,
    },
    {
        "id": "find_hide",
        "prefix": "{A} found the {adj} {noun} {prep} the {surface}. {B} hid it again.",
        "suffix": "{B} put the {noun} {prep} the {surface}.",
        "neg_prefix": "{A} did not find the {adj} {noun} {prep} the {surface}. {B} hid it again.",
        "neg_suffix": "{B} did not put the {noun} {prep} the {surface}.",
        "ops": ("rename", "swap", "place", "negate"),
        "has_c": False,
        "place": ("on", "under"),
    },
    {
        "id": "pack_ship",
        "prefix": "{A} packed {num} {noun_plural} into the {surface}.",
        "suffix": "{B} shipped {num} {noun_plural} from the {surface}.",
        "ops": ("rename", "number"),
        "has_c": False,
        "number": ("two", "three"),
    },
    {
        "id": "promise_give",
        "prefix": "{A} promised {B} a {adj} {noun}.",
        "suffix": "{B} gave the {noun} to {C}.",
        "neg_prefix": "{A} did not promise {B} a {adj} {noun}.",
        "neg_suffix": "{B} did not give the {noun} to {C}.",
        "ops": ("rename", "swap", "negate"),
        "has_c": True,
    },
]

ALL_TEMPLATES = list(TRAIN_TEMPLATES) + list(EXTRA_TEMPLATES) + list(BROAD_TEMPLATES)


def _combos(ops):
    for r in range(1, min(3, len(ops)) + 1):
        for combo in itertools.combinations(ops, r):
            yield list(combo)


def record_any(rng, names, template, ops, index, split="train"):
    for _ in range(64):
        slots = _slots(rng, names, template)
        factual_prefix, factual_suffix, _ = render(template, slots, [])
        edited_prefix, edited_suffix, phrases = render(template, slots, ops)
        if edited_suffix == factual_suffix or edited_prefix == factual_prefix:
            continue
        return {
            "id": f"{split}-{template['id']}-{index:05d}-{'-'.join(ops)}",
            "split": split,
            "template_id": template["id"],
            "operations": list(ops),
            "composition": len(ops) > 1,
            "instruction": _instruction(phrases),
            "prefix": factual_prefix,
            "edited_prefix": edited_prefix,
            "suffix": edited_suffix,
            "factual_suffix": factual_suffix,
            "entities": {
                "A": slots["A"], "B": slots["B"], "C": slots["C"],
                "A2": slots["A2"], "B2": slots["B2"], "C2": slots["C2"],
            },
            "op_vector": _op_vector(ops),
        }
    raise ValueError(f"could not build {template['id']}/{ops}")


def _dedup(records):
    seen = set()
    out = []
    for r in records:
        key = (r["instruction"], r["prefix"], r["edited_prefix"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _build(templates, names, n_per_combo, seed, split):
    rng = random.Random(seed)
    out = []
    for template in templates:
        for ops in _combos(template["ops"]):
            for i in range(n_per_combo):
                try:
                    out.append(record_any(rng, names, template, ops, i, split))
                except ValueError:
                    continue
    return _dedup(out)


def build_train(n_per_combo: int = 300, seed: int = 46):
    return _build(ALL_TEMPLATES, NAMES_TRAIN, n_per_combo, seed, "train")


def build_dev(n_per_combo: int = 80, seed: int = 999):
    out = _build(ALL_TEMPLATES, NAMES_EVAL, n_per_combo, seed, "dev")
    random.Random(seed + 1).shuffle(out)
    return out
