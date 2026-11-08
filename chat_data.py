"""KALIA-Chat v0: from-scratch synthetic instruction/dialogue data.

Every record is generated from templates, a hand-written word list, a
hand-written fact table and rule-based answers. No external instruction
dataset, no model-generated responses, no distillation.
"""

from __future__ import annotations

import random
import re
from typing import Any

import torch

PAD_ID = 50256
NEWLINE_ID = 198
EOS_ID = 50256
MAX_LEN = 192

# ---------------------------------------------------------------------------
# hand-written vocabulary and facts
# ---------------------------------------------------------------------------
WORDS = [
    "cat", "dog", "sun", "moon", "star", "tree", "book", "lamp", "fish",
    "bird", "hand", "door", "wind", "rain", "snow", "fire", "rock", "sand",
    "gold", "leaf", "milk", "cake", "song", "game", "home", "road", "city",
    "farm", "ship", "king", "queen", "house", "water", "light", "night",
    "green", "black", "white", "happy", "small", "large", "quick", "slow",
    "jump", "walk", "read", "write", "speak", "listen", "think", "learn",
    "teach", "build", "make", "play", "work", "help", "find", "open",
    "close", "start", "stop", "turn", "move", "look", "hear", "feel",
    "know", "want", "need", "give", "take", "come", "tell", "ask",
    "answer", "question", "story", "word", "letter", "number", "color",
    "shape", "animal", "plant", "river", "mountain", "ocean", "planet",
    "world", "music", "sound", "voice", "smile", "laugh", "dream", "hope",
    "peace", "brave", "kind", "smart", "strong", "young", "good", "best",
    "fast", "warm", "cold", "soft", "hard", "bright", "dark", "clean",
    "quiet", "loud", "round", "square",
]

COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "black", "white"]
ANIMALS = ["cat", "dog", "bird", "fish", "horse", "rabbit", "tiger", "mouse"]
ADJECTIVES = ["small", "happy", "quick", "brave", "kind", "bright", "quiet", "strong"]
NAMES = ["Sam", "Ada", "Mira", "Ravi", "Lina", "Omar", "Zoe", "Kai", "Nora",
         "Leo", "Isha", "Noah", "Ava", "Rin", "Tara", "Dev", "Maya", "Arun",
         "Sara", "Nikhil"]

FACTS: list[dict[str, str]] = [
    {"fact": "The sky is blue.", "question": "What color is the sky?", "answer": "blue"},
    {"fact": "Grass is green.", "question": "What color is grass?", "answer": "green"},
    {"fact": "The sun is yellow.", "question": "What color is the sun?", "answer": "yellow"},
    {"fact": "Snow is white.", "question": "What color is snow?", "answer": "white"},
    {"fact": "Coal is black.", "question": "What color is coal?", "answer": "black"},
    {"fact": "A triangle has three sides.", "question": "How many sides does a triangle have?", "answer": "three"},
    {"fact": "A square has four sides.", "question": "How many sides does a square have?", "answer": "four"},
    {"fact": "A circle has no straight sides.", "question": "How many straight sides does a circle have?", "answer": "zero"},
    {"fact": "A pentagon has five sides.", "question": "How many sides does a pentagon have?", "answer": "five"},
    {"fact": "A cat is an animal.", "question": "Is a cat an animal?", "answer": "yes"},
    {"fact": "A dog is an animal.", "question": "Is a dog an animal?", "answer": "yes"},
    {"fact": "A bird is an animal.", "question": "Is a bird an animal?", "answer": "yes"},
    {"fact": "A fish is an animal.", "question": "Is a fish an animal?", "answer": "yes"},
    {"fact": "A cat says meow.", "question": "What does a cat say?", "answer": "meow"},
    {"fact": "A dog says woof.", "question": "What does a dog say?", "answer": "woof"},
    {"fact": "A cow says moo.", "question": "What does a cow say?", "answer": "moo"},
    {"fact": "A bird says tweet.", "question": "What does a bird say?", "answer": "tweet"},
    {"fact": "The capital of France is Paris.", "question": "What is the capital of France?", "answer": "Paris"},
    {"fact": "The capital of Japan is Tokyo.", "question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"fact": "The capital of Italy is Rome.", "question": "What is the capital of Italy?", "answer": "Rome"},
    {"fact": "The capital of India is New Delhi.", "question": "What is the capital of India?", "answer": "New Delhi"},
    {"fact": "The capital of Egypt is Cairo.", "question": "What is the capital of Egypt?", "answer": "Cairo"},
    {"fact": "The capital of Canada is Ottawa.", "question": "What is the capital of Canada?", "answer": "Ottawa"},
    {"fact": "The capital of Brazil is Brasilia.", "question": "What is the capital of Brazil?", "answer": "Brasilia"},
    {"fact": "The capital of Australia is Canberra.", "question": "What is the capital of Australia?", "answer": "Canberra"},
    {"fact": "Water is a liquid.", "question": "Is water a liquid?", "answer": "yes"},
    {"fact": "Ice is a solid.", "question": "Is ice a solid?", "answer": "yes"},
    {"fact": "The sun is a star.", "question": "Is the sun a star?", "answer": "yes"},
    {"fact": "Earth is a planet.", "question": "Is Earth a planet?", "answer": "yes"},
    {"fact": "Plants need water.", "question": "What do plants need?", "answer": "water"},
    {"fact": "Humans need oxygen.", "question": "What do humans need?", "answer": "oxygen"},
    {"fact": "A week has seven days.", "question": "How many days are in a week?", "answer": "seven"},
    {"fact": "A year has twelve months.", "question": "How many months are in a year?", "answer": "twelve"},
    {"fact": "A day has twenty-four hours.", "question": "How many hours are in a day?", "answer": "twenty-four"},
    {"fact": "A human has two eyes.", "question": "How many eyes does a human have?", "answer": "two"},
    {"fact": "A spider has eight legs.", "question": "How many legs does a spider have?", "answer": "eight"},
    {"fact": "A dog has four legs.", "question": "How many legs does a dog have?", "answer": "four"},
    {"fact": "The alphabet has twenty-six letters.", "question": "How many letters are in the alphabet?", "answer": "twenty-six"},
    {"fact": "The Nile is a river.", "question": "Is the Nile a river?", "answer": "yes"},
    {"fact": "Mount Everest is a mountain.", "question": "Is Mount Everest a mountain?", "answer": "yes"},
    {"fact": "The Pacific is an ocean.", "question": "Is the Pacific an ocean?", "answer": "yes"},
]


def _split(items: list, seed: int) -> tuple[list, list, list]:
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


WORD_SPLITS = dict(zip(("train", "val", "test"), _split(WORDS, 56)))
FACT_SPLITS = dict(zip(("train", "val", "test"), _split(FACTS, 57)))
NAME_SPLITS = dict(zip(("train", "val", "test"), _split(NAMES, 58)))

# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------
ADD_TRAIN = ["What is {a} + {b}?", "Add {a} and {b}.",
             "What is the sum of {a} and {b}?"]
ADD_TEST = ["Calculate {a} plus {b}.", "{a} + {b} = ?"]
SUB_TRAIN = ["What is {a} - {b}?", "Subtract {b} from {a}.",
             "What is the difference between {a} and {b}?"]
SUB_TEST = ["Calculate {a} minus {b}.", "{a} - {b} = ?"]

WORD_OPS = {
    "reverse": {
        "train": ["Reverse the word {word}.", "Write {word} backwards."],
        "test": ["What is {word} reversed?"],
        "fn": lambda w: w[::-1],
    },
    "count": {
        "train": ["How many letters are in {word}?", "Count the letters in {word}."],
        "test": ["What is the length of {word}?"],
        "fn": lambda w: str(len(w)),
    },
    "uppercase": {
        "train": ["Write {word} in uppercase.", "Make {word} uppercase."],
        "test": ["Convert {word} to capital letters."],
        "fn": lambda w: w.upper(),
    },
    "first": {
        "train": ["What is the first letter of {word}?", "Which letter starts {word}?"],
        "test": ["What letter does {word} begin with?"],
        "fn": lambda w: w[0],
    },
    "last": {
        "train": ["What is the last letter of {word}?", "Which letter ends {word}?"],
        "test": ["What letter does {word} end with?"],
        "fn": lambda w: w[-1],
    },
    "repeat": {
        "train": ["Repeat {word} three times.", "Say {word} three times."],
        "test": ["Write {word} three times."],
        "fn": lambda w: f"{w} {w} {w}",
    },
}

CONTEXT_TRAIN = ["Context: {fact} Question: {question}",
                 "Read this: {fact} Now answer: {question}"]
CONTEXT_TEST = ["Fact: {fact} Q: {question}"]

INSTRUCTION = {
    "list_colors": {
        "train": ["List three colors.", "Name three colors."],
        "test": ["Give me three colors."],
    },
    "list_animals": {
        "train": ["List two animals.", "Name two animals."],
        "test": ["Give me two animals."],
    },
    "numbers": {
        "train": ["Write the numbers from 1 to 5.", "Count from 1 to 5."],
        "test": ["List the numbers 1 through 5."],
    },
    "alphabetical": {
        "train": ["Put these words in alphabetical order: {words}.",
                  "Sort these words: {words}."],
        "test": ["Arrange these words alphabetically: {words}."],
    },
    "uppercase": {
        "train": ["Write {word} in uppercase.", "Make {word} uppercase."],
        "test": ["Convert {word} to capital letters."],
    },
    "sentence": {
        "train": ["Write a short sentence about a {animal}.",
                  "Say something about a {animal}."],
        "test": ["Write one sentence about a {animal}."],
    },
}

CHAT_PAIRS = [
    {"user": "Hello", "responses": ["Hello! How can I help?", "Hi! What can I do for you?"],
     "keywords": ["hello", "hi"]},
    {"user": "Hi", "responses": ["Hi! What can I do for you?", "Hello! How can I help?"],
     "keywords": ["hi", "hello"]},
    {"user": "How are you?", "responses": ["I'm a small model, but I'm ready to help."],
     "keywords": ["small", "ready", "help"]},
    {"user": "What is your name?", "responses": ["I am KALIA, a small from-scratch model."],
     "keywords": ["kalia"]},
    {"user": "What can you do?", "responses": ["I can do simple arithmetic, word tasks, and short instructions."],
     "keywords": ["arithmetic", "word", "instruction", "help"]},
    {"user": "Thank you", "responses": ["You're welcome.", "You are welcome."],
     "keywords": ["welcome"]},
    {"user": "Thanks", "responses": ["You're welcome.", "Happy to help."],
     "keywords": ["welcome", "help"]},
    {"user": "Goodbye", "responses": ["Goodbye!", "Bye! Come back soon."],
     "keywords": ["bye", "goodbye"]},
    {"user": "See you", "responses": ["See you!", "Goodbye!"],
     "keywords": ["see", "bye", "goodbye"]},
    {"user": "Good morning", "responses": ["Good morning! How can I help?"],
     "keywords": ["morning", "help"]},
]

REFUSAL_PAIRS = [
    {"user": "What is the capital of Germany?", "responses": ["I don't know."]},
    {"user": "What is the capital of Spain?", "responses": ["I don't know."]},
    {"user": "Can you book a flight?", "responses": ["I can't do that."]},
    {"user": "Can you order food for me?", "responses": ["I can't do that."]},
    {"user": "Tell me a secret.", "responses": ["I can't help with that."]},
    {"user": "What is the meaning of life?", "responses": ["I don't know."]},
    {"user": "Give me medical advice.", "responses": ["I can't give medical advice."]},
    {"user": "Who will win the election?", "responses": ["I don't know."]},
    {"user": "Can you drive a car?", "responses": ["I can't do that."]},
    {"user": "What is my password?", "responses": ["I don't know."]},
]

# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------
def _rec(task: str, user: str, assistant: str, *, answer: str | None = None,
         check: str = "exact", **meta: Any) -> dict:
    return {"task": task, "user": user, "assistant": assistant,
            "answer": answer if answer is not None else assistant,
            "check": check, "meta": meta}


def _turns_rec(task: str, turns: list[dict], answer: str, **meta: Any) -> dict:
    return {"task": task, "turns": turns, "answer": answer,
            "check": "exact", "meta": meta}


def gen_arithmetic(n: int, rng: random.Random, split: str) -> list[dict]:
    out = []
    add_t = ADD_TRAIN if split != "test" else ADD_TEST
    sub_t = SUB_TRAIN if split != "test" else SUB_TEST
    for _ in range(n):
        if rng.random() < 0.5:
            a, b = rng.randint(0, 20), rng.randint(0, 20)
            user = rng.choice(add_t).format(a=a, b=b)
            ans = str(a + b)
        else:
            a = rng.randint(0, 20)
            b = rng.randint(0, a)
            user = rng.choice(sub_t).format(a=a, b=b)
            ans = str(a - b)
        out.append(_rec("arithmetic", user, ans, answer=ans))
    return out


def gen_word(n: int, rng: random.Random, split: str) -> list[dict]:
    out = []
    words = WORD_SPLITS[split]
    ops = list(WORD_OPS)
    for _ in range(n):
        op = rng.choice(ops)
        spec = WORD_OPS[op]
        word = rng.choice(words)
        user = rng.choice(spec["test"] if split == "test" else spec["train"]).format(word=word)
        ans = spec["fn"](word)
        out.append(_rec("word", user, ans, answer=ans, subtask=op, word=word))
    return out


def gen_context(n: int, rng: random.Random, split: str) -> list[dict]:
    out = []
    facts = FACT_SPLITS[split]
    templates = CONTEXT_TEST if split == "test" else CONTEXT_TRAIN
    for _ in range(n):
        f = rng.choice(facts)
        user = rng.choice(templates).format(fact=f["fact"], question=f["question"])
        out.append(_rec("context_qa", user, f["answer"], answer=f["answer"]))
    return out


def gen_instruction(n: int, rng: random.Random, split: str) -> list[dict]:
    out = []
    words = WORD_SPLITS[split]
    for _ in range(n):
        sub = rng.choice(list(INSTRUCTION))
        spec = INSTRUCTION[sub]
        template = rng.choice(spec["test"] if split == "test" else spec["train"])
        animal = ""
        if sub == "list_colors":
            chosen = rng.sample(COLORS, 3)
            user = template
            ans = ", ".join(chosen)
        elif sub == "list_animals":
            chosen = rng.sample(ANIMALS, 2)
            user = template
            ans = ", ".join(chosen)
        elif sub == "numbers":
            user = template
            ans = "1, 2, 3, 4, 5"
        elif sub == "alphabetical":
            chosen = rng.sample(words, 3)
            user = template.format(words=", ".join(chosen))
            ans = ", ".join(sorted(chosen))
        elif sub == "uppercase":
            word = rng.choice(words)
            user = template.format(word=word)
            ans = word.upper()
        else:  # sentence
            animal = rng.choice(ANIMALS)
            adj = rng.choice(ADJECTIVES)
            user = template.format(animal=animal)
            ans = f"The {animal} is {adj}."
        out.append(_rec("instruction", user, ans, answer=ans, subtask=sub,
                        animal=animal))
    return out


def gen_chat(n: int, rng: random.Random, split: str) -> list[dict]:
    out = []
    # split the pair list by index, deterministic
    pairs = CHAT_PAIRS[:]
    random.Random(59).shuffle(pairs)
    n_train = int(len(pairs) * 0.8)
    n_val = int(len(pairs) * 0.1)
    pool = (pairs[:n_train] if split == "train"
            else pairs[n_train:n_train + n_val] if split == "val"
            else pairs[n_train + n_val:])
    if not pool:
        pool = pairs
    for _ in range(n):
        p = rng.choice(pool)
        out.append(_rec("chat", p["user"], rng.choice(p["responses"]),
                        answer=p["responses"][0], check="keyword",
                        keywords=p["keywords"]))
    return out


def gen_refusal(n: int, rng: random.Random, split: str) -> list[dict]:
    out = []
    pairs = REFUSAL_PAIRS[:]
    random.Random(60).shuffle(pairs)
    n_train = int(len(pairs) * 0.8)
    n_val = int(len(pairs) * 0.1)
    pool = (pairs[:n_train] if split == "train"
            else pairs[n_train:n_train + n_val] if split == "val"
            else pairs[n_train + n_val:])
    if not pool:
        pool = pairs
    for _ in range(n):
        p = rng.choice(pool)
        out.append(_rec("refusal", p["user"], rng.choice(p["responses"]),
                        answer=p["responses"][0], check="refusal"))
    return out


def gen_multi(n: int, rng: random.Random, split: str) -> list[dict]:
    out = []
    names = NAME_SPLITS[split]
    for _ in range(n):
        kind = rng.choice(["name", "count", "chain"])
        if kind == "name":
            name = rng.choice(names)
            turns = [{"user": f"My name is {name}.", "assistant": f"Hello, {name}."},
                     {"user": "What is my name?", "assistant": name}]
            ans = name
        elif kind == "count":
            k = rng.randint(1, 9)
            turns = [{"user": f"I have {k} apples.", "assistant": "Okay."},
                     {"user": "How many apples do I have?", "assistant": str(k)}]
            ans = str(k)
        else:
            a, b = rng.randint(0, 10), rng.randint(0, 10)
            d = rng.randint(1, 5)
            turns = [{"user": f"What is {a} + {b}?", "assistant": str(a + b)},
                     {"user": f"Add {d} to that.", "assistant": str(a + b + d)}]
            ans = str(a + b + d)
        out.append(_turns_rec("multi_turn", turns, ans, kind=kind))
    return out


def build_splits(seed: int = 56, n_train: int = 100_000, n_val: int = 8_000,
                 n_test: int = 8_000) -> dict[str, list[dict]]:
    """Build deterministic train/val/test splits."""
    rng = random.Random(seed)
    result = {}
    for split, total in (("train", n_train), ("val", n_val), ("test", n_test)):
        n_arith = int(total * 0.25)
        n_word = int(total * 0.25)
        n_context = int(total * 0.20)
        n_instr = int(total * 0.15)
        n_chat = int(total * 0.05)
        n_ref = int(total * 0.05)
        n_multi = total - n_arith - n_word - n_context - n_instr - n_chat - n_ref
        recs = (gen_arithmetic(n_arith, rng, split)
                + gen_word(n_word, rng, split)
                + gen_context(n_context, rng, split)
                + gen_instruction(n_instr, rng, split)
                + gen_chat(n_chat, rng, split)
                + gen_refusal(n_ref, rng, split)
                + gen_multi(n_multi, rng, split))
        rng.shuffle(recs)
        result[split] = recs
    return result


# ---------------------------------------------------------------------------
# chat template / dataset
# ---------------------------------------------------------------------------
def _turns(record: dict) -> list[dict]:
    return record.get("turns") or [{"user": record["user"],
                                    "assistant": record["assistant"]}]


def chat_text(record: dict) -> str:
    return "".join(f"User: {t['user']}\nAssistant: {t['assistant']}\n"
                   for t in _turns(record))


def chat_prompt(record: dict) -> str:
    turns = _turns(record)
    parts = [f"User: {t['user']}\nAssistant: {t['assistant']}\n"
             for t in turns[:-1]]
    parts.append(f"User: {turns[-1]['user']}\nAssistant: ")
    return "".join(parts)


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], enc, max_len: int = MAX_LEN):
        self.items: list[tuple[list[int], list[int], list[bool]]] = []
        for r in records:
            ids: list[int] = []
            mask: list[bool] = []
            for t in _turns(r):
                u = enc.encode_ordinary(f"User: {t['user']}\nAssistant: ")
                a = enc.encode_ordinary(f"{t['assistant']}\n")
                ids.extend(u); mask.extend([False] * len(u))
                ids.extend(a); mask.extend([True] * len(a))
            if len(ids) > max_len:
                ids = ids[-max_len:]; mask = mask[-max_len:]
            self.items.append((ids[:-1], ids[1:], mask[1:]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


def collate(items, max_len: int = MAX_LEN):
    b = len(items)
    x = torch.full((b, max_len), PAD_ID, dtype=torch.long)
    y = torch.full((b, max_len), -100, dtype=torch.long)
    lm = torch.zeros((b, max_len), dtype=torch.bool)
    tm = torch.zeros((b, max_len), dtype=torch.bool)
    for i, (ids, labels, mask) in enumerate(items):
        n = min(len(ids), max_len)
        x[i, :n] = torch.tensor(ids[:n])
        y[i, :n] = torch.tensor(labels[:n])
        lm[i, :n] = torch.tensor(mask[:n])
        tm[i, :n] = True
    causal = torch.ones(max_len, max_len, dtype=torch.bool).tril()
    attn = causal.unsqueeze(0).unsqueeze(0) & tm.unsqueeze(1).unsqueeze(2)
    return x, y, lm, tm, attn


# ---------------------------------------------------------------------------
# generation / checks
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@torch.no_grad()
def generate_response(model, record: dict, enc, device, max_new: int = 24) -> str:
    was_training = model.training
    model.eval()
    ids = enc.encode_ordinary(chat_prompt(record))
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
    if was_training:
        model.train()
    text = normalize(enc.decode(out))
    if "User:" in text:
        text = text.split("User:")[0].strip()
    return text


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def check_instruction(record: dict, response: str) -> bool:
    sub = record.get("meta", {}).get("subtask")
    r = _norm(response)
    if sub == "list_colors":
        toks = set(re.findall(r"[a-z]+", r))
        return len(toks & set(COLORS)) >= 3
    if sub == "list_animals":
        toks = set(re.findall(r"[a-z]+", r))
        return len(toks & set(ANIMALS)) >= 2
    if sub == "numbers":
        return re.findall(r"\d+", r) == ["1", "2", "3", "4", "5"]
    if sub == "alphabetical":
        expected = re.findall(r"[a-z]+", _norm(record["answer"]))
        toks = re.findall(r"[a-z]+", r)
        return toks[-len(expected):] == expected if len(toks) >= len(expected) else False
    if sub == "uppercase":
        return response.strip().rstrip(".") == record["answer"]
    if sub == "sentence":
        animal = record.get("meta", {}).get("animal", "")
        return bool(animal) and animal in r and len(r.split()) >= 4
    return False


def check_record(record: dict, response: str) -> bool:
    task = record["task"]
    r = _norm(response)
    if task == "arithmetic":
        return r.rstrip(".").strip() == _norm(record["answer"])
    if task == "word":
        if record.get("meta", {}).get("subtask") == "uppercase":
            return response.strip().rstrip(".") == record["answer"]
        return r.rstrip(".").strip() == _norm(record["answer"])
    if task == "context_qa":
        return r.rstrip(".").strip() == _norm(record["answer"])
    if task == "instruction":
        return check_instruction(record, response)
    if task == "chat":
        if not response.strip():
            return False
        if r == _norm(record["user"]):
            return False
        kws = record.get("meta", {}).get("keywords") or []
        return any(k in r for k in kws) if kws else True
    if task == "refusal":
        return any(p in r for p in ("don't know", "do not know", "can't", "cannot",
                                    "can not", "unable", "not able"))
    if task == "multi_turn":
        return r.rstrip(".").strip() == _norm(record["answer"])
    return False


def format_ok(record: dict, response: str) -> bool:
    if not response.strip():
        return False
    if _norm(response) == _norm(record.get("user", "")):
        return False
    return len(response.split()) <= 40
