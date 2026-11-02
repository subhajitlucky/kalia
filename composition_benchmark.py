"""X40 follow-up: a clean 2x2 composition benchmark.

X40's unseen-composition set mixed two variables:

1. a **new combination** of interventions (single -> pair), and
2. a **new sentence type / causal relation** (the `pack_box` "packed -> the
   bag holds" template never appears in training).

`number+place` therefore scored 0.0 in 9/10 arms for a reason that is not
composition. This benchmark separates the two:

|  | single change (seen) | pair change (unseen) |
|---|---|---|
| **seen sentence type** | S-S | S-P |
| **unseen sentence type** | U-S | U-P |

- **S-S** is the easy baseline: a train template, one intervention.
- **S-P** is the real composition test: a train template and causal relation,
  but two interventions. Nothing else is new.
- **U-S** isolates template/lexical generalization.
- **U-P** is both at once (the old X40 aggregate).

All four cells use the **same new name set**, so name novelty is constant.
The benchmark is a diagnostic, not a re-score of X40's registered bars. It
must not be used to reopen the X40 verdict.

Usage:
    python composition_benchmark.py --out-dir out/x40-bench --seed 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from intervention_pairs import (
    CHUNK,
    COMPOSITIONS,
    EVAL_TEMPLATES,
    NAMES_EVAL,
    NAMES_TRAIN,
    TRAIN_TEMPLATES,
    _record,
)

# Deliberately disjoint from both X40 name sets. The same set is used in every
# cell, so a cell difference cannot be a name-novelty difference.
NAMES_BENCH = [
    "Bella", "Caleb", "Dina", "Eli", "Fiona", "Gabe", "Hana", "Ivan",
    "Jade", "Kofi", "Luna", "Mateo", "Nadia", "Pablo", "Quinn", "Rosa",
    "Sven", "Tara", "Umar", "Willa",
]

CELLS = (
    ("seen", "single"),
    ("seen", "pair"),
    ("unseen", "single"),
    ("unseen", "pair"),
)
CELL_ID = {
    ("seen", "single"): "S-S",
    ("seen", "pair"): "S-P",
    ("unseen", "single"): "U-S",
    ("unseen", "pair"): "U-P",
}


def _valid_option(template: dict, ops: list[str]) -> bool:
    """Reject a degenerate composition before it enters the cell.

    `give_keep` + swap+negate is the motivating case: after the swap, negation
    makes the *original* giver keep the object, which is exactly the factual
    suffix. The candidates collapse, and the cell would measure nothing.
    """
    try:
        _record(random.Random(0), NAMES_BENCH, template, ops, "eval", 0)
    except ValueError:
        return False
    return True


def _options(templates: list[dict], pair: bool) -> list[tuple[dict, list[str]]]:
    """Template/operation combinations available in this cell."""
    out: list[tuple[dict, list[str]]] = []
    if pair:
        for template in templates:
            for comp in COMPOSITIONS:
                if all(op in template["ops"] for op in comp):
                    if _valid_option(template, list(comp)):
                        out.append((template, list(comp)))
    else:
        for template in templates:
            for op in template["ops"]:
                out.append((template, [op]))
    return out


def generate_benchmark(seed: int = 2026, n_per_cell: int = 400) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    for familiarity, composition in CELLS:
        templates = TRAIN_TEMPLATES if familiarity == "seen" else EVAL_TEMPLATES
        options = _options(templates, pair=(composition == "pair"))
        if not options:
            raise ValueError(f"no options for cell {familiarity}/{composition}")
        cell = CELL_ID[(familiarity, composition)]
        for i in range(n_per_cell):
            template, ops = options[i % len(options)]
            record = _record(rng, NAMES_BENCH, template, ops, "eval", i)
            record["cell"] = cell
            record["template_familiarity"] = familiarity
            record["composition_type"] = composition
            records.append(record)
    return records


def _token_lengths(texts: list[str]) -> int:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    return max(len(enc.encode_ordinary(text)) for text in texts) if texts else 0


def audit(records: list[dict]) -> dict:
    cells = {cell: [r for r in records if r["cell"] == cell]
             for cell in CELL_ID.values()}
    assert set(cells) == set(CELL_ID.values()), sorted(cells)
    assert all(len(rows) > 0 for rows in cells.values())
    assert all(r["template_id"] in {t["id"] for t in TRAIN_TEMPLATES}
               for r in records if r["template_familiarity"] == "seen")
    assert all(r["template_id"] in {t["id"] for t in EVAL_TEMPLATES}
               for r in records if r["template_familiarity"] == "unseen")
    assert all(r["n_way"] == 2 for r in records if r["composition_type"] == "single")
    assert all(r["n_way"] == 4 for r in records if r["composition_type"] == "pair")
    assert all(len(r["candidates"]) == r["n_way"] for r in records)
    assert all(len(set(r["candidates"])) == len(r["candidates"]) for r in records)
    assert all(r["candidates"][r["correct_index"]] == r["suffix"] for r in records)
    assert all(r["suffix"] != r["factual_suffix"] for r in records)
    assert all(r["edited_prefix"] != r["prefix"] for r in records)
    names = {v for r in records for v in r["entities"].values() if v}
    assert names <= set(NAMES_BENCH), sorted(names - set(NAMES_BENCH))
    assert not names & set(NAMES_TRAIN), "benchmark leaks a train name"
    assert not names & set(NAMES_EVAL), "benchmark leaks an X40 eval name"
    # Every cell draws from the same pool, so a cell difference is not a
    # name-novelty difference.
    per_cell_names = {
        cell: {v for r in rows for v in r["entities"].values() if v}
        for cell, rows in cells.items()
    }
    union = set().union(*per_cell_names.values())
    assert union == names
    for cell, cell_names in per_cell_names.items():
        assert cell_names, cell

    lengths = {
        "prompt_tokens": _token_lengths([f"{r['instruction']} {r['prefix']}" for r in records]),
        "suffix_tokens": _token_lengths([r["suffix"] for r in records]),
        "edited_prefix_tokens": _token_lengths([r["edited_prefix"] for r in records]),
    }
    assert lengths["prompt_tokens"] <= CHUNK, lengths
    assert lengths["suffix_tokens"] <= CHUNK, lengths
    assert lengths["edited_prefix_tokens"] <= CHUNK, lengths

    def counts(rows):
        out: dict[str, int] = {}
        for r in rows:
            out["+".join(r["operations"])] = out.get("+".join(r["operations"]), 0) + 1
        return dict(sorted(out.items()))

    return {
        "cells": {
            cell: {
                "n": len(rows),
                "n_way": rows[0]["n_way"],
                "chance": 1.0 / rows[0]["n_way"],
                "counts": counts(rows),
            }
            for cell, rows in sorted(cells.items())
        },
        "names": sorted(names),
        "lengths": lengths,
        "examples": {
            cell: {
                "instruction": rows[0]["instruction"],
                "prefix": rows[0]["prefix"],
                "correct": rows[0]["suffix"],
                "candidates": rows[0]["candidates"],
            }
            for cell, rows in sorted(cells.items())
        },
    }


def write_benchmark(out_dir: Path, seed: int = 2026, n_per_cell: int = 400) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = generate_benchmark(seed, n_per_cell)
    path = out_dir / "bench.jsonl"
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")
    manifest = audit(records)
    manifest.update({
        "seed": seed,
        "n": len(records),
        "n_per_cell": n_per_cell,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
    (out_dir / "audit.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
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
        "cells": manifest["cells"],
        "lengths": manifest["lengths"],
    }, indent=2))


if __name__ == "__main__":
    main()
