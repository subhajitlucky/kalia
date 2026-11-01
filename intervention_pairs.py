"""X40: intervention pairs for a causal event workspace.

Each example is a small causal chain. The model sees an **intervention
instruction** and the **factual prefix**; the target is the **edited
continuation** (the consequence in the intervened world). The edited prefix is
kept only as the target for the auxiliary next-state loss, never shown to the
model.

Train and eval use disjoint templates and names. Training contains **single**
interventions only. Evaluation contains singles plus four **unseen
compositions**: rename+swap, swap+negate, number+place, rename+negate. The
`pairs` arm trains on the same text as ordinary LM data; the `IP` arm adds the
event workspace and the next-state loss.

Usage:
    python intervention_pairs.py --out-dir out/ip --seed 1338
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
CHUNK = 64

OP_NAMES = ("rename", "swap", "negate", "number", "place")
OP_DIM = len(OP_NAMES) + 1  # one multi-hot over ops + a composition flag

NAMES_TRAIN = [
    "Maya", "Leo", "Nora", "Sam", "Aria", "Finn", "Iris", "Milo",
    "Zara", "Owen", "Lena", "Theo", "Ruby", "Hugo", "Ella", "Max",
    "Nina", "Otis", "Cleo", "Rhys", "Ada", "Kai", "Vera", "Sol",
]
NAMES_EVAL = [
    "Tessa", "Rowan", "Priya", "Diego", "Freya", "Anton",
    "Lila", "Marco", "Sana", "Elise", "Omar", "June",
]
ADJS = ["red", "old", "small", "shiny"]
NOUNS = ["book", "apple", "stone", "flower", "coin", "letter", "lantern", "ribbon"]
PLURALS = {
    "book": "books", "apple": "apples", "stone": "stones", "flower": "flowers",
    "coin": "coins", "letter": "letters", "lantern": "lanterns", "ribbon": "ribbons",
}
SURFACES = ["table", "shelf", "bench", "floor", "desk", "ledge"]
LOCATIONS = ["market", "garden", "harbor", "village", "station", "library"]

# Template fields:
#   prefix/suffix  — the factual causal chain
#   neg_prefix/    — the negated chain, if the template supports `negate`
#   neg_suffix
#   ops            — the interventions the template supports
#   has_c          — the chain has a third participant
#   number         — (original, edited) if the template supports `number`
#   place          — (original, edited) if the template supports `place`
TRAIN_TEMPLATES = [
    {
        "id": "give_keep",
        "prefix": "{A} gave {B} the {adj} {noun}.",
        "suffix": "{B} kept the {noun}.",
        "neg_prefix": "{A} did not give {B} the {adj} {noun}.",
        "neg_suffix": "{A} kept the {noun}.",
        "ops": ("rename", "swap", "negate"),
        "has_c": False,
    },
    {
        "id": "hide_find",
        "prefix": "{A} hid the {adj} {noun} {prep} the {surface}. {B} looked for it.",
        "suffix": "{B} found it {prep} the {surface}.",
        "neg_prefix": "{A} did not hide the {adj} {noun}. {B} looked for it.",
        "neg_suffix": "{B} did not find it.",
        "ops": ("rename", "swap", "place", "negate"),
        "has_c": False,
        "place": ("on", "under"),
    },
    {
        "id": "put_count",
        "prefix": "{A} put {num} {noun_plural} {prep} the {surface}.",
        "suffix": "There are {num} {noun_plural} {prep} the {surface}.",
        "ops": ("number", "place"),
        "has_c": False,
        "number": ("two", "three"),
        "place": ("on", "under"),
    },
    {
        "id": "meet_tell",
        "prefix": "{A} met {B} at the {location}. {A} told {B} a secret.",
        "suffix": "{B} told {C} the secret.",
        "neg_prefix": "{A} met {B} at the {location}. {A} did not tell {B} a secret.",
        "neg_suffix": "{B} did not know the secret.",
        "ops": ("rename", "swap", "negate"),
        "has_c": True,
    },
]

EVAL_TEMPLATES = [
    {
        "id": "tell_secret",
        "prefix": "{A} told {B} a secret. {B} promised to keep it.",
        "suffix": "{B} told {C} the secret.",
        "ops": ("rename", "swap"),
        "has_c": True,
    },
    {
        "id": "ask_help",
        "prefix": "{A} asked {B} for the {noun}. {B} gave it to {A}.",
        "suffix": "{A} thanked {B}.",
        "neg_prefix": "{A} asked {B} for the {noun}. {B} did not give it to {A}.",
        "neg_suffix": "{A} did not thank {B}.",
        "ops": ("rename", "swap", "negate"),
        "has_c": False,
    },
    {
        "id": "pack_box",
        "prefix": "{A} packed {num} {noun_plural} {prep} the {surface}.",
        "suffix": "The bag holds {num} {noun_plural} {prep} the {surface}.",
        "ops": ("number", "place"),
        "has_c": False,
        "number": ("two", "three"),
        "place": ("on", "under"),
    },
]

COMPOSITIONS = [
    ("rename", "swap"),
    ("swap", "negate"),
    ("number", "place"),
    ("rename", "negate"),
]


def _join(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _slots(rng: random.Random, names: list[str], template: dict) -> dict:
    """Draw disjoint entity names and the template's optional values."""
    need_c = bool(template.get("has_c"))
    picked = rng.sample(names, 3 if need_c else 2)
    a, b = picked[0], picked[1]
    c = picked[2] if need_c else ""
    remaining = [n for n in names if n not in picked]
    picked2 = rng.sample(remaining, 3 if need_c else 2)
    a2, b2 = picked2[0], picked2[1]
    c2 = picked2[2] if need_c else ""
    noun = rng.choice(NOUNS)
    return {
        "A": a, "B": b, "C": c,
        "A2": a2, "B2": b2, "C2": c2,
        "adj": rng.choice(ADJS),
        "noun": noun,
        "noun_plural": PLURALS[noun],
        "prep": template.get("place", ("on", "under"))[0],
        "surface": rng.choice(SURFACES),
        "location": rng.choice(LOCATIONS),
        "num": template.get("number", ("two", "three"))[0],
    }


def _apply(template: dict, slots: dict, ops: list[str]) -> tuple[dict, list[str]]:
    """Apply interventions in order and return the instruction phrases."""
    s = dict(slots)
    phrases: list[str] = []
    for op in ops:
        if op == "rename":
            pairs = [(k, s[k], s[k + "2"]) for k in ("A", "B", "C") if s.get(k)]
            phrases.append("rename " + _join([f"{old} to {new}" for _, old, new in pairs]))
            for key, _, new in pairs:
                s[key] = new
        elif op == "swap":
            old_a, old_b = s["A"], s["B"]
            phrases.append(f"swap {old_a} and {old_b}")
            s["A"], s["B"] = old_b, old_a
        elif op == "negate":
            if template.get("neg_prefix") is None:
                raise ValueError(f"{template['id']} does not support negate")
            phrases.append("negate the event")
        elif op == "number":
            if template.get("number") is None:
                raise ValueError(f"{template['id']} does not support number")
            alt = template["number"][1]
            phrases.append(f"change {s['num']} to {alt}")
            s["num"] = alt
        elif op == "place":
            if template.get("place") is None:
                raise ValueError(f"{template['id']} does not support place")
            alt = template["place"][1]
            phrases.append(f"change {s['prep']} to {alt}")
            s["prep"] = alt
        else:
            raise ValueError(f"unknown intervention {op!r}")
    return s, phrases


def render(template: dict, slots: dict, ops: list[str]) -> tuple[str, str, list[str]]:
    """Render the factual or edited chain. Returns (prefix, suffix, phrases)."""
    s, phrases = _apply(template, slots, ops)
    neg = "negate" in ops
    prefix = template["neg_prefix" if neg else "prefix"].format(**s)
    suffix = template["neg_suffix" if neg else "suffix"].format(**s)
    return prefix, suffix, phrases


def _instruction(phrases: list[str]) -> str:
    return "Intervention: " + "; ".join(phrases) + "."


def _op_vector(ops: list[str]) -> list[float]:
    vec = [1.0 if op in ops else 0.0 for op in OP_NAMES]
    vec.append(1.0 if len(ops) > 1 else 0.0)
    return vec


def _candidates(rng: random.Random, template: dict, slots: dict,
                ops: list[str]) -> tuple[list[str], list[str]]:
    """Correct suffix, then the 2-way or 4-way alternatives."""
    if len(ops) == 1:
        correct = render(template, slots, ops)[1]
        factual = render(template, slots, [])[1]
        return [correct, factual], [correct]
    if len(ops) != 2:
        raise ValueError(f"expected one or two interventions, got {ops}")
    correct = render(template, slots, ops)[1]
    factual = render(template, slots, [])[1]
    first = render(template, slots, [ops[0]])[1]
    second = render(template, slots, [ops[1]])[1]
    return [correct, factual, first, second], [correct]


def _record(
    rng: random.Random,
    names: list[str],
    template: dict,
    ops: list[str],
    split: str,
    index: int,
) -> dict:
    """Build one example, retrying if random slots make two candidates equal."""
    for _ in range(32):
        slots = _slots(rng, names, template)
        factual_prefix, factual_suffix, _ = render(template, slots, [])
        edited_prefix, edited_suffix, phrases = render(template, slots, ops)
        if edited_suffix == factual_suffix or edited_prefix == factual_prefix:
            continue
        candidates, correct = _candidates(rng, template, slots, ops)
        if len(set(candidates)) != len(candidates):
            continue
        if split == "eval":
            rng.shuffle(candidates)
            correct_index = candidates.index(correct[0])
            n_way = len(candidates)
        else:
            candidates = [edited_suffix]
            correct_index = 0
            n_way = 1
        text_fields = (factual_prefix, edited_prefix, edited_suffix, *candidates)
        if any("{" in text or "}" in text for text in text_fields):
            raise ValueError(f"unfilled slot for {template['id']}/{ops}")
        if any(token in text for text in (factual_prefix, edited_prefix) for token in ("  ", " .")):
            raise ValueError(f"malformed punctuation for {template['id']}/{ops}")
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
            "candidates": candidates,
            "correct_index": correct_index,
            "n_way": n_way,
            "op_vector": _op_vector(ops),
            "entities": {
                "A": slots["A"], "B": slots["B"], "C": slots["C"],
                "A2": slots["A2"], "B2": slots["B2"], "C2": slots["C2"],
            },
        }
    raise ValueError(f"could not build a distinct example for {template['id']}/{ops}")


def generate(
    seed: int = 1338,
    n_train: int = 1000,
    n_eval: int = 150,
    n_comp: int = 150,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    train: list[dict] = []
    eval_: list[dict] = []
    for template in TRAIN_TEMPLATES:
        for i in range(n_train):
            op = rng.choice(template["ops"])
            train.append(_record(rng, NAMES_TRAIN, template, [op], "train", i))
    for template in EVAL_TEMPLATES:
        for i in range(n_eval):
            op = rng.choice(template["ops"])
            eval_.append(_record(rng, NAMES_EVAL, template, [op], "eval", i))
    for comp in COMPOSITIONS:
        templates = [t for t in EVAL_TEMPLATES if all(op in t["ops"] for op in comp)]
        if not templates:
            raise ValueError(f"no eval template supports {comp}")
        for i in range(n_comp):
            template = rng.choice(templates)
            eval_.append(_record(rng, NAMES_EVAL, template, list(comp), "eval", i))
    return train, eval_


def _token_lengths(texts: list[str]) -> int:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    return max(len(enc.encode_ordinary(text)) for text in texts) if texts else 0


def audit(train: list[dict], eval_: list[dict]) -> dict:
    train_templates = sorted({r["template_id"] for r in train})
    eval_templates = sorted({r["template_id"] for r in eval_})
    train_names = sorted({v for r in train for v in r["entities"].values() if v})
    eval_names = sorted({v for r in eval_ for v in r["entities"].values() if v})
    assert train_templates and eval_templates
    assert not (set(train_templates) & set(eval_templates)), "template leakage"
    assert not (set(train_names) & set(eval_names)), "name leakage"
    assert all(not r["composition"] for r in train), "train has a composition"
    assert all(r["n_way"] in (1, 2, 4) for r in train + eval_)
    assert all(len(set(r["candidates"])) == len(r["candidates"]) for r in eval_)
    assert all(r["suffix"] != r["factual_suffix"] for r in train + eval_)
    assert all(r["edited_prefix"] != r["prefix"] for r in train + eval_)
    assert all(len(r["op_vector"]) == OP_DIM for r in train + eval_)
    assert all(r["candidates"][r["correct_index"]] == r["suffix"] for r in eval_)
    assert all(len(r["candidates"]) == r["n_way"] for r in eval_)
    assert all(r["n_way"] == 1 for r in train)

    train_prompts = [f"{r['instruction']} {r['prefix']}" for r in train]
    eval_prompts = [f"{r['instruction']} {r['prefix']}" for r in eval_]
    train_suffixes = [r["suffix"] for r in train]
    eval_suffixes = [r["suffix"] for r in eval_]
    lengths = {
        "train_prompt_tokens": _token_lengths(train_prompts),
        "eval_prompt_tokens": _token_lengths(eval_prompts),
        "train_suffix_tokens": _token_lengths(train_suffixes),
        "eval_suffix_tokens": _token_lengths(eval_suffixes),
        "train_edited_prefix_tokens": _token_lengths([r["edited_prefix"] for r in train]),
        "eval_edited_prefix_tokens": _token_lengths([r["edited_prefix"] for r in eval_]),
    }
    assert lengths["train_prompt_tokens"] <= CHUNK, lengths
    assert lengths["eval_prompt_tokens"] <= CHUNK, lengths
    assert lengths["train_suffix_tokens"] <= CHUNK, lengths
    assert lengths["eval_suffix_tokens"] <= CHUNK, lengths
    assert lengths["train_edited_prefix_tokens"] <= CHUNK, lengths
    assert lengths["eval_edited_prefix_tokens"] <= CHUNK, lengths

    def counts(rows):
        out: dict[str, int] = {}
        for r in rows:
            key = "+".join(r["operations"])
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    examples = []
    seen = set()
    for r in eval_:
        key = "+".join(r["operations"])
        if key in seen:
            continue
        seen.add(key)
        examples.append({
            "operations": r["operations"],
            "instruction": r["instruction"],
            "prefix": r["prefix"],
            "correct": r["suffix"],
            "alternatives": [c for i, c in enumerate(r["candidates"]) if i != r["correct_index"]],
        })

    return {
        "train_templates": train_templates,
        "eval_templates": eval_templates,
        "train_names": train_names,
        "eval_names": eval_names,
        "train_counts": counts(train),
        "eval_counts": counts(eval_),
        "lengths": lengths,
        "examples": examples,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pairs(out_dir: Path, seed: int = 1338, n_train: int = 1000,
                n_eval: int = 150, n_comp: int = 150) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train, eval_ = generate(seed, n_train, n_eval, n_comp)
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
class IPBatch:
    x: torch.Tensor
    y: torch.Tensor
    loss_mask: torch.Tensor
    token_mask: torch.Tensor
    attn_mask: torch.Tensor
    target_x: torch.Tensor
    target_mask: torch.Tensor
    target_attn_mask: torch.Tensor
    op_vec: torch.Tensor


def _pad_left(ids: list[int], length: int) -> list[int]:
    if len(ids) > length:
        raise ValueError(f"sequence of {len(ids)} tokens exceeds {length}")
    return [PAD_ID] * (length - len(ids)) + ids


def _pad_right(ids: list[int], length: int) -> list[int]:
    if len(ids) > length:
        raise ValueError(f"sequence of {len(ids)} tokens exceeds {length}")
    return ids + [PAD_ID] * (length - len(ids))


def causal_pad_mask(token_mask: torch.Tensor) -> torch.Tensor:
    """Causal attention with padded keys blocked. ``True`` means attend."""
    t = token_mask.size(1)
    causal = torch.ones(t, t, dtype=torch.bool, device=token_mask.device).tril()
    return causal.unsqueeze(0).unsqueeze(0) & token_mask.unsqueeze(1).unsqueeze(2)


class InterventionDataset:
    """Fixed ``prefix_len``-token prompt + ``suffix_len``-token continuation.

    The prompt is left-padded so the last prompt token sits at the chunk
    boundary. The suffix is right-padded. ``loss_mask`` supervises only the
    suffix; ``token_mask`` excludes padding from the workspace write and from
    attention keys. ``target_x`` is the edited prefix, used only by the
    counterfactual next-state loss.
    """

    def __init__(self, path: str | Path, prefix_len: int = CHUNK,
                 suffix_len: int = CHUNK):
        import tiktoken

        self.path = Path(path)
        self.prefix_len = int(prefix_len)
        self.suffix_len = int(suffix_len)
        self.records = [
            json.loads(line) for line in self.path.read_text().splitlines() if line.strip()
        ]
        if not self.records:
            raise ValueError(f"no records in {self.path}")
        self.enc = tiktoken.get_encoding("gpt2")
        self._cache: dict[tuple[int, str | None], tuple] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _ids(self, text: str, limit: int) -> list[int]:
        # Do not silently truncate: a cut instruction or suffix changes the
        # example. The padding helpers raise instead, and the audit pins the
        # generated data below the chunk size.
        ids = self.enc.encode_ordinary(text)
        if len(ids) > limit:
            raise ValueError(
                f"text needs {len(ids)} tokens but the limit is {limit}: {text!r}"
            )
        return ids if ids else [PAD_ID]

    def _encode(self, prompt: str, suffix: str, edited_prefix: str,
                op_vector: list[float]) -> tuple:
        p_ids = self._ids(prompt, self.prefix_len)
        s_ids = self._ids(suffix, self.suffix_len)
        t_ids = self._ids(edited_prefix, self.prefix_len)
        prefix = _pad_left(p_ids, self.prefix_len)
        suffix = _pad_right(s_ids, self.suffix_len)
        target = _pad_left(t_ids, self.prefix_len)
        x = prefix + suffix
        y = x[1:] + [PAD_ID]
        t = self.prefix_len + self.suffix_len
        loss_mask = torch.zeros(t, dtype=torch.bool)
        loss_mask[self.prefix_len - 1:self.prefix_len + len(s_ids) - 1] = True
        token_mask = torch.zeros(t, dtype=torch.bool)
        token_mask[self.prefix_len - len(p_ids):self.prefix_len] = True
        token_mask[self.prefix_len:self.prefix_len + len(s_ids)] = True
        target_mask = torch.zeros(self.prefix_len, dtype=torch.bool)
        target_mask[self.prefix_len - len(t_ids):] = True
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            loss_mask,
            token_mask,
            torch.tensor(target, dtype=torch.long),
            target_mask,
            torch.tensor(op_vector, dtype=torch.float32),
        )

    def example(self, index: int) -> tuple:
        r = self.records[index]
        key = (index, None)
        if key not in self._cache:
            self._cache[key] = self._encode(
                f"{r['instruction']} {r['prefix']}", r["suffix"],
                r["edited_prefix"], r["op_vector"],
            )
        return self._cache[key]

    def candidate(self, index: int, candidate: str) -> tuple:
        """Encode one candidate continuation for multiple-choice scoring."""
        r = self.records[index]
        key = (index, candidate)
        if key not in self._cache:
            s_ids = self._ids(candidate, self.suffix_len)
            p_ids = self._ids(f"{r['instruction']} {r['prefix']}", self.prefix_len)
            prefix = _pad_left(p_ids, self.prefix_len)
            suffix = _pad_right(s_ids, self.suffix_len)
            x = prefix + suffix
            y = x[1:] + [PAD_ID]
            t = self.prefix_len + self.suffix_len
            loss_mask = torch.zeros(t, dtype=torch.bool)
            loss_mask[self.prefix_len - 1:self.prefix_len + len(s_ids) - 1] = True
            token_mask = torch.zeros(t, dtype=torch.bool)
            token_mask[self.prefix_len - len(p_ids):self.prefix_len] = True
            token_mask[self.prefix_len:self.prefix_len + len(s_ids)] = True
            self._cache[key] = (
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                loss_mask, token_mask,
            )
        return self._cache[key]

    def _collate(self, rows: list[tuple]) -> IPBatch:
        x, y, loss_mask, token_mask, target_x, target_mask, op_vec = zip(*rows)
        x = torch.stack(x)
        y = torch.stack(y)
        loss_mask = torch.stack(loss_mask)
        token_mask = torch.stack(token_mask)
        target_mask = torch.stack(target_mask)
        return IPBatch(
            x=x, y=y, loss_mask=loss_mask, token_mask=token_mask,
            attn_mask=causal_pad_mask(token_mask),
            target_x=torch.stack(target_x), target_mask=target_mask,
            target_attn_mask=causal_pad_mask(target_mask),
            op_vec=torch.stack(op_vec),
        )

    def get_batch(self, batch_size: int, device: torch.device,
                  generator: torch.Generator | None = None) -> IPBatch:
        idx = torch.randint(len(self.records), (batch_size,), generator=generator).tolist()
        batch = self._collate([self.example(i) for i in idx])
        return _to_device(batch, device)


def _to_device(batch, device):
    return type(batch)(**{
        name: getattr(batch, name).to(device) for name in batch.__dataclass_fields__
    })


def candidate_items(records: list[dict]) -> list[tuple[int, int]]:
    """(record index, candidate index) for every eval candidate."""
    return [
        (i, j) for i, r in enumerate(records) for j in range(len(r["candidates"]))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1338)
    parser.add_argument("--n-train", type=int, default=1000, help="pairs per train template")
    parser.add_argument("--n-eval", type=int, default=150, help="singles per eval template")
    parser.add_argument("--n-comp", type=int, default=150, help="pairs per composition")
    args = parser.parse_args()
    manifest = write_pairs(args.out_dir, args.seed, args.n_train, args.n_eval, args.n_comp)
    keep = (
        "seed", "n_train", "n_eval", "train_templates", "eval_templates",
        "train_counts", "eval_counts", "lengths",
    )
    print(json.dumps({k: manifest[k] for k in keep}, indent=2))


if __name__ == "__main__":
    main()
