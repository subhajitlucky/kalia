"""X38: deterministic counterfactual pairs for the residual bottleneck.

Two kinds of pair:

- **invariant** — the meaning is preserved by a consistent entity rename or an
  adjective synonym. The stable representation should not move.
- **sensitive** — the meaning changes by a subject/object swap, a negation, a
  number change or a spatial change. The stable representation should move.

The generator is rule-based on purpose. It is auditable, deterministic and
cheap; the interesting variable is the mechanism, not an LLM-written dataset.
Train and eval use disjoint template IDs and disjoint name sets, so the C-3
probe is held out.

Usage:
    python crb_pairs.py --out-dir out/crb --seed 1338
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

PAD_ID = 50256  # GPT-2 EOS, used only as padding; masks exclude it.

NAMES_TRAIN = [
    "Maya", "Leo", "Nora", "Sam", "Aria", "Finn", "Iris", "Milo",
    "Zara", "Owen", "Lena", "Theo", "Ruby", "Hugo", "Ella", "Max",
    "Nina", "Otis", "Cleo", "Rhys", "Ada", "Kai", "Vera", "Sol",
]
NAMES_EVAL = [
    "Tessa", "Rowan", "Priya", "Diego", "Freya", "Anton",
    "Lila", "Marco", "Sana", "Elise", "Omar", "June",
]

# A small, unambiguous synonym map. If an adjective is absent, the template
# simply does not emit a synonym pair.
ADJECTIVES = {
    "red": "crimson",
    "small": "tiny",
    "old": "ancient",
    "shiny": "gleaming",
}

NOUNS = ["book", "apple", "stone", "flower", "coin", "letter", "lantern", "ribbon"]
PLURALS = {
    "book": "books", "apple": "apples", "stone": "stones", "flower": "flowers",
    "coin": "coins", "letter": "letters", "lantern": "lanterns", "ribbon": "ribbons",
}
PLACES = ["market", "garden", "harbor", "village", "station", "library"]
SURFACES = ["table", "shelf", "bench", "floor", "desk", "ledge"]

# Template fields:
#   base    — the original sentence (may use {A}, {B}, {adj}, {noun}, {place},
#             {surface}, {num}, {noun_plural}, {prep})
#   swap    — role swap, if supported
#   negate  — negation, if supported
#   number  — number-change template, if supported; original uses "two",
#             edited uses "three"
#   place   — (original_prep, edited_prep), if supported; base must use {prep}
#   synonym — True if the synonym edit applies to {adj}
TEMPLATES_TRAIN = [
    {
        "id": "give",
        "base": "{A} gave {B} {art} {adj} {noun}.",
        "swap": "{B} gave {A} {art} {adj} {noun}.",
        "negate": "{A} did not give {B} {art} {adj} {noun}.",
        "number": "{A} gave {B} {num} {noun_plural}.",
        "synonym": True,
    },
    {
        "id": "meet",
        "base": "{A} met {B} {prep} the {place}.",
        "swap": "{B} met {A} {prep} the {place}.",
        "negate": "{A} did not meet {B} {prep} the {place}.",
        "place": ("at", "near"),
        "prep": "at",
    },
    {
        "id": "tell",
        "base": "{A} told {B} a story.",
        "swap": "{B} told {A} a story.",
        "negate": "{A} did not tell {B} a story.",
    },
    {
        "id": "walk",
        "base": "{A} and {B} walked {prep} the {place}.",
        "negate": "{A} and {B} did not walk {prep} the {place}.",
        "place": ("to", "away from"),
        "prep": "to",
    },
    {
        "id": "find",
        "base": "{A} found {art} {adj} {noun} near {B}.",
        "swap": "{B} found {art} {adj} {noun} near {A}.",
        "negate": "{A} did not find {art} {adj} {noun} near {B}.",
        "synonym": True,
    },
    {
        "id": "put",
        "base": "{A} put the {noun} {prep} the {surface}.",
        "negate": "{A} did not put the {noun} {prep} the {surface}.",
        "place": ("on", "under"),
        "prep": "on",
    },
]

TEMPLATES_EVAL = [
    {
        "id": "ask",
        "base": "{A} asked {B} for the {adj} {noun}.",
        "swap": "{B} asked {A} for the {adj} {noun}.",
        "negate": "{A} did not ask {B} for the {adj} {noun}.",
        "synonym": True,
    },
    {
        "id": "see",
        "base": "{A} saw {B} {prep} the {surface}.",
        "swap": "{B} saw {A} {prep} the {surface}.",
        "negate": "{A} did not see {B} {prep} the {surface}.",
        "place": ("behind", "in front of"),
        "prep": "behind",
    },
]

INVARIANT_SUBTYPES = ("rename", "synonym")
SENSITIVE_SUBTYPES = ("swap", "negate", "number", "place")


def _slots(rng: random.Random, names: list[str], template: dict) -> dict:
    a, b = rng.sample(names, 2)
    a2, b2 = rng.sample([n for n in names if n not in (a, b)], 2)
    adj = rng.choice(list(ADJECTIVES))
    noun = rng.choice(NOUNS)
    return {
        "A": a, "B": b, "A2": a2, "B2": b2,
        "adj": adj, "adj_syn": ADJECTIVES[adj],
        "art": "an" if adj[0].lower() in "aeiou" else "a",
        "art_syn": "an" if ADJECTIVES[adj][0].lower() in "aeiou" else "a",
        "noun": noun, "noun_plural": PLURALS[noun],
        "place": rng.choice(PLACES),
        "surface": rng.choice(SURFACES),
        "num": "two",
        "prep": template.get("prep", "at"),
    }


def _subtypes(template: dict) -> list[str]:
    out = ["rename"]
    if template.get("synonym"):
        out.append("synonym")
    for key in ("swap", "negate", "number", "place"):
        if key in template:
            out.append(key)
    return out


def _make_pair(
    rng: random.Random,
    names: list[str],
    template: dict,
    subtype: str,
    split: str,
    index: int,
) -> dict:
    s = _slots(rng, names, template)
    if subtype == "rename":
        original = template["base"].format(**s)
        edited = template["base"].format(**{**s, "A": s["A2"], "B": s["B2"]})
        edit_type = "invariant"
    elif subtype == "synonym":
        original = template["base"].format(**s)
        edited = template["base"].format(**{**s, "adj": s["adj_syn"], "art": s["art_syn"]})
        edit_type = "invariant"
    elif subtype == "swap":
        original = template["base"].format(**s)
        edited = template["swap"].format(**s)
        edit_type = "sensitive"
    elif subtype == "negate":
        original = template["base"].format(**s)
        edited = template["negate"].format(**s)
        edit_type = "sensitive"
    elif subtype == "number":
        original = template["number"].format(**s)
        edited = template["number"].format(**{**s, "num": "three"})
        edit_type = "sensitive"
    elif subtype == "place":
        orig_prep, edit_prep = template["place"]
        original = template["base"].format(**{**s, "prep": orig_prep})
        edited = template["base"].format(**{**s, "prep": edit_prep})
        edit_type = "sensitive"
    else:
        raise ValueError(f"unknown subtype {subtype!r} for template {template['id']!r}")

    if original == edited:
        raise ValueError(f"empty edit: {template['id']}/{subtype}")
    if subtype == "rename":
        detail = {"A": s["A"], "B": s["B"], "A2": s["A2"], "B2": s["B2"]}
    elif subtype == "synonym":
        detail = {"adj": s["adj"], "synonym": s["adj_syn"]}
    elif subtype == "swap":
        detail = {"roles": "A/B -> B/A"}
    elif subtype == "negate":
        detail = {"polarity": "affirmative -> negative"}
    elif subtype == "number":
        detail = {"number": "two -> three"}
    else:
        detail = {"prep": f"{template['place'][0]} -> {template['place'][1]}"}
    return {
        "id": f"{split}-{template['id']}-{index:05d}-{subtype}",
        "split": split,
        "template_id": template["id"],
        "edit_type": edit_type,
        "subtype": subtype,
        "original": original,
        "edited": edited,
        "edit_detail": detail,
        "entities": {"A": s["A"], "B": s["B"], "A2": s["A2"], "B2": s["B2"]},
    }


def generate(
    seed: int = 1338,
    n_train: int = 1200,
    n_eval: int = 200,
) -> tuple[list[dict], list[dict]]:
    """Generate disjoint train and eval pairs.

    ``n_train``/``n_eval`` are per template. Subtypes are cycled, not sampled,
    so every split gets an approximately equal mix.
    """
    rng = random.Random(seed)
    train: list[dict] = []
    eval_: list[dict] = []
    for templates, names, split, count, sink in (
        (TEMPLATES_TRAIN, NAMES_TRAIN, "train", n_train, train),
        (TEMPLATES_EVAL, NAMES_EVAL, "eval", n_eval, eval_),
    ):
        for template in templates:
            subtypes = _subtypes(template)
            for i in range(count):
                sink.append(_make_pair(rng, names, template, subtypes[i % len(subtypes)], split, i))
    return train, eval_


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def audit(train: list[dict], eval_: list[dict]) -> dict:
    train_templates = sorted({r["template_id"] for r in train})
    eval_templates = sorted({r["template_id"] for r in eval_})
    train_names = sorted({v for r in train for v in r["entities"].values() if v})
    eval_names = sorted({v for r in eval_ for v in r["entities"].values() if v})
    assert not (set(train_templates) & set(eval_templates)), "template leakage"
    assert not (set(train_names) & set(eval_names)), "name leakage"
    assert all(r["edit_type"] in ("invariant", "sensitive") for r in train + eval_)
    assert all(r["subtype"] in INVARIANT_SUBTYPES + SENSITIVE_SUBTYPES for r in train + eval_)
    assert all(r["original"] != r["edited"] for r in train + eval_)

    def counts(rows):
        out: dict[str, int] = {}
        for row in rows:
            key = f"{row['edit_type']}/{row['subtype']}"
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    examples = []
    seen = set()
    for row in train + eval_:
        if row["subtype"] in seen:
            continue
        seen.add(row["subtype"])
        examples.append({
            "subtype": row["subtype"],
            "original": row["original"],
            "edited": row["edited"],
        })
    return {
        "train_templates": train_templates,
        "eval_templates": eval_templates,
        "train_names": train_names,
        "eval_names": eval_names,
        "train_counts": counts(train),
        "eval_counts": counts(eval_),
        "examples": examples,
    }


def write_pairs(out_dir: Path, seed: int = 1338, n_train: int = 1200, n_eval: int = 200) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train, eval_ = generate(seed=seed, n_train=n_train, n_eval=n_eval)
    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    train_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in train) + "\n")
    eval_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in eval_) + "\n")
    manifest = audit(train, eval_)
    manifest.update({
        "seed": seed,
        "n_train": len(train),
        "n_eval": len(eval_),
        "train_sha256": _sha256(train_path),
        "eval_sha256": _sha256(eval_path),
    })
    (out_dir / "audit.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


@dataclass
class CFBatch:
    orig: torch.Tensor
    edit: torch.Tensor
    types: torch.Tensor
    orig_mask: torch.Tensor
    edit_mask: torch.Tensor


class CounterfactualPairDataset:
    """Tokenised pairs with balanced invariant/sensitive sampling."""

    def __init__(self, path: str | Path, max_len: int = 128, split: str | None = None):
        import tiktoken

        self.path = Path(path)
        self.max_len = int(max_len)
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        if split is not None:
            rows = [r for r in rows if r["split"] == split]
        if not rows:
            raise ValueError(f"no rows in {self.path} for split={split!r}")
        self.records = rows
        self.enc = tiktoken.get_encoding("gpt2")
        self._orig = [self._encode(r["original"]) for r in rows]
        self._edit = [self._encode(r["edited"]) for r in rows]
        self._inv = [i for i, r in enumerate(rows) if r["edit_type"] == "invariant"]
        self._sen = [i for i, r in enumerate(rows) if r["edit_type"] == "sensitive"]
        if not self._inv or not self._sen:
            raise ValueError("the pair file must contain both invariant and sensitive pairs")

    def _encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.enc.encode_ordinary(text)[: self.max_len]
        if not ids:
            ids = [PAD_ID]
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[: len(ids)] = True
        out = torch.full((self.max_len,), PAD_ID, dtype=torch.long)
        out[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out, mask

    def __len__(self) -> int:
        return len(self.records)

    def get_batch(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> CFBatch:
        n_inv = batch_size // 2
        n_sen = batch_size - n_inv
        inv = [self._inv[i] for i in torch.randint(len(self._inv), (n_inv,), generator=generator).tolist()]
        sen = [self._sen[i] for i in torch.randint(len(self._sen), (n_sen,), generator=generator).tolist()]
        idx = inv + sen
        perm = torch.randperm(len(idx), generator=generator).tolist()
        idx = [idx[i] for i in perm]
        orig = torch.stack([self._orig[i][0] for i in idx]).to(device)
        edit = torch.stack([self._edit[i][0] for i in idx]).to(device)
        orig_mask = torch.stack([self._orig[i][1] for i in idx]).to(device)
        edit_mask = torch.stack([self._edit[i][1] for i in idx]).to(device)
        types = torch.tensor(
            [0 if self.records[i]["edit_type"] == "invariant" else 1 for i in idx],
            dtype=torch.long,
            device=device,
        )
        return CFBatch(orig, edit, types, orig_mask, edit_mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1338)
    parser.add_argument("--n-train", type=int, default=1200, help="pairs per train template")
    parser.add_argument("--n-eval", type=int, default=200, help="pairs per eval template")
    args = parser.parse_args()
    manifest = write_pairs(args.out_dir, args.seed, args.n_train, args.n_eval)
    print(json.dumps({k: manifest[k] for k in ("seed", "n_train", "n_eval",
                                               "train_templates", "eval_templates",
                                               "train_counts", "eval_counts")}, indent=2))


if __name__ == "__main__":
    main()
