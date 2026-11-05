"""X44/X45 fresh benchmark: new names, new templates, 1/2/3 operations.

The templates and names are disjoint from every training/eval set used in
X40–X43. The 3-op cell is the X45 depth test; it is 8-way (correct, factual,
three singles, three pairs).

Usage:
    python fresh_benchmark.py --out-dir out/x44-fresh --seed 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from intervention_pairs import (
    NAMES_EVAL,
    NAMES_TRAIN,
    TRAIN_TEMPLATES,
    EVAL_TEMPLATES,
    _instruction,
    _op_vector,
    _record,
    _slots,
    render,
)
from composition_benchmark import NAMES_BENCH
from edit_compiler import EXTRA_TEMPLATES

FRESH_NAMES = [
    "Anya", "Bodhi", "Cyrus", "Dara", "Eshan", "Farah", "Gita", "Hiro",
    "Indra", "Jaya", "Kiran", "Leela", "Mira", "Naveen", "Oona", "Pavel",
    "Rhea", "Sami", "Veer", "Zoya",
]

FRESH_TEMPLATES = [
    {
        "id": "bake_share",
        "prefix": "{A} baked a {adj} {noun} for {B}.",
        "suffix": "{B} shared the {noun} with {C}.",
        "neg_prefix": "{A} did not bake a {adj} {noun} for {B}.",
        "neg_suffix": "{B} did not share the {noun}.",
        "ops": ("rename", "swap", "negate"),
        "has_c": True,
    },
    {
        "id": "find_return",
        "prefix": "{A} found the {adj} {noun} {prep} the {surface}. {B} returned it to {A}.",
        "suffix": "{A} thanked {B} for the {noun} {prep} the {surface}.",
        "neg_prefix": "{A} did not find the {adj} {noun} {prep} the {surface}. {B} looked for it.",
        "neg_suffix": "{B} did not return it.",
        "ops": ("rename", "swap", "place", "negate"),
        "has_c": False,
        "place": ("on", "under"),
    },
    {
        "id": "count_pack",
        "prefix": "{A} put {num} {noun_plural} {prep} the {surface}.",
        "suffix": "There are {num} {noun_plural} {prep} the {surface}.",
        "ops": ("number", "place"),
        "has_c": False,
        "number": ("two", "three"),
        "place": ("on", "under"),
    },
    {
        "id": "promise_tell",
        "prefix": "{A} promised {B} a {adj} {noun}. {B} told {C} about it.",
        "suffix": "{C} asked {B} for the {noun}.",
        "neg_prefix": "{A} did not promise {B} a {adj} {noun}. {B} told {C} about it.",
        "neg_suffix": "{C} did not ask {B} for the {noun}.",
        "ops": ("rename", "swap", "negate"),
        "has_c": True,
    },
]

CELLS = ("F-S", "F-P", "F-T")


def _valid_option(template, ops) -> bool:
    try:
        if len(ops) == 3:
            _record3(random.Random(0), FRESH_NAMES, template, ops, 0)
        else:
            _record(random.Random(0), FRESH_NAMES, template, ops, "eval", 0)
    except ValueError:
        return False
    return True


def _options(templates, ops_per: int):
    out = []
    for template in templates:
        if ops_per == 1:
            for op in template["ops"]:
                out.append((template, [op]))
        elif ops_per == 2:
            for i, a in enumerate(template["ops"]):
                for b in template["ops"][i + 1:]:
                    out.append((template, [a, b]))
        elif ops_per == 3:
            for i, a in enumerate(template["ops"]):
                for j, b in enumerate(template["ops"][i + 1:], i + 1):
                    for c in template["ops"][j + 1:]:
                        out.append((template, [a, b, c]))
        else:
            raise ValueError(ops_per)
    return [(t, ops) for t, ops in out if _valid_option(t, ops)]


def _record3(rng, names, template, ops, index):
    """8-way record: correct, factual, each single, each pair."""
    for _ in range(64):
        slots = _slots(rng, names, template)
        factual_prefix, factual_suffix, _ = render(template, slots, [])
        edited_prefix, edited_suffix, phrases = render(template, slots, ops)
        if edited_suffix == factual_suffix or edited_prefix == factual_prefix:
            continue
        candidates = [edited_suffix, factual_suffix]
        for op in ops:
            candidates.append(render(template, slots, [op])[1])
        for i, a in enumerate(ops):
            for b in ops[i + 1:]:
                candidates.append(render(template, slots, [a, b])[1])
        if len(set(candidates)) != len(candidates):
            continue
        rng.shuffle(candidates)
        return {
            "id": f"eval-{template['id']}-{index:05d}-{'-'.join(ops)}",
            "split": "eval",
            "template_id": template["id"],
            "operations": list(ops),
            "composition": True,
            "instruction": _instruction(phrases),
            "prefix": factual_prefix,
            "edited_prefix": edited_prefix,
            "suffix": edited_suffix,
            "factual_suffix": factual_suffix,
            "candidates": candidates,
            "correct_index": candidates.index(edited_suffix),
            "n_way": len(candidates),
            "op_vector": _op_vector(ops),
            "entities": {
                "A": slots["A"], "B": slots["B"], "C": slots["C"],
                "A2": slots["A2"], "B2": slots["B2"], "C2": slots["C2"],
            },
        }
    raise ValueError(f"could not build distinct 3-op example for {template['id']}/{ops}")


def generate(seed: int = 2026, n_per_cell: int = 400) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for cell, ops_per in (("F-S", 1), ("F-P", 2), ("F-T", 3)):
        options = _options(FRESH_TEMPLATES, ops_per)
        if not options:
            raise ValueError(f"no options for {cell}")
        for i in range(n_per_cell):
            template, ops = options[i % len(options)]
            if ops_per == 3:
                rec = _record3(rng, FRESH_NAMES, template, ops, i)
            else:
                rec = _record(rng, FRESH_NAMES, template, ops, "eval", i)
            rec["cell"] = cell
            rec["template_familiarity"] = "fresh"
            rec["composition_type"] = "single" if ops_per == 1 else (
                "pair" if ops_per == 2 else "triple"
            )
            records.append(rec)
    return records


def audit(records: list[dict]) -> dict:
    forbidden_names = set(NAMES_TRAIN) | set(NAMES_EVAL) | set(NAMES_BENCH)
    forbidden_templates = (
        {t["id"] for t in TRAIN_TEMPLATES}
        | {t["id"] for t in EVAL_TEMPLATES}
        | {t["id"] for t in EXTRA_TEMPLATES}
    )
    names = {v for r in records for v in r["entities"].values() if v}
    templates = {r["template_id"] for r in records}
    assert names <= set(FRESH_NAMES), sorted(names - set(FRESH_NAMES))
    assert not (names & forbidden_names), sorted(names & forbidden_names)
    assert not (templates & forbidden_templates), sorted(templates & forbidden_templates)
    assert all(len(set(r["candidates"])) == len(r["candidates"]) for r in records)
    assert all(r["candidates"][r["correct_index"]] == r["suffix"] for r in records)
    assert all(r["suffix"] != r["factual_suffix"] for r in records)
    assert all(r["edited_prefix"] != r["prefix"] for r in records)
    cells = {cell: [r for r in records if r["cell"] == cell] for cell in CELLS}
    assert all(cells.values())
    for cell, rows in cells.items():
        assert len(rows) == len({r["id"] for r in rows}), cell
        n_way = rows[0]["n_way"]
        assert all(r["n_way"] == n_way for r in rows)
    return {
        "cells": {
            cell: {
                "n": len(rows),
                "n_way": rows[0]["n_way"],
                "chance": 1.0 / rows[0]["n_way"],
                "counts": {
                    "+".join(r["operations"]): sum(
                        1 for x in rows if x["operations"] == r["operations"]
                    )
                    for r in rows
                },
            }
            for cell, rows in cells.items()
        },
        "names": sorted(names),
        "templates": sorted(templates),
        "examples": {
            cell: {
                "instruction": rows[0]["instruction"],
                "prefix": rows[0]["prefix"],
                "correct": rows[0]["suffix"],
                "candidates": rows[0]["candidates"],
            }
            for cell, rows in cells.items()
        },
    }


def write_benchmark(out_dir: Path, seed: int = 2026, n_per_cell: int = 400) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = generate(seed, n_per_cell)
    path = out_dir / "bench.jsonl"
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")
    manifest = audit(records)
    manifest.update({
        "seed": seed,
        "n": len(records),
        "n_per_cell": n_per_cell,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
    (out_dir / "audit.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-per-cell", type=int, default=400)
    args = parser.parse_args()
    manifest = write_benchmark(args.out_dir, args.seed, args.n_per_cell)
    print(json.dumps({
        "seed": manifest["seed"],
        "n": manifest["n"],
        "sha256": manifest["sha256"],
        "cells": manifest["cells"],
        "names": manifest["names"],
        "templates": manifest["templates"],
    }, indent=2))


if __name__ == "__main__":
    main()
