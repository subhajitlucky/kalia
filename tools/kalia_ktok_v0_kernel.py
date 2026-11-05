"""Kaggle CPU kernel: K-Tok v0.

Decodes a sample of the training corpus (GPT-2 BPE is lossless), trains
byte-level BPE tokenizers at 32k/50k/64k plus a digit-aware 32k variant, and
compares them with GPT-2 BPE on held-out text, code, and numbers.

No model training. No GPU. Outputs tokenizer JSONs + report.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

TRAIN_TOKENS = 50_000_000
VAL_TOKENS = 2_000_000
VOCABS = (32000, 50000, 64000)

# --- data --------------------------------------------------------------------
train_hits = sorted(Path("/kaggle/input").rglob("train.bin"))
preferred = [p for p in train_hits if "kalia-prep-v2b" in str(p)]
assert preferred, f"kalia-prep-v2b not attached or lacks train.bin: {train_hits}"
train_bin = preferred[0]
val_bin = train_bin.parent / "val.bin"
assert val_bin.exists(), f"val.bin missing beside {train_bin}"
print("train:", train_bin, "val:", val_bin, flush=True)

try:
    import tiktoken
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"],
                   check=True)
    import tiktoken

try:
    import tokenizers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"],
                   check=True)
    import tokenizers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

print("tokenizers", tokenizers.__version__, flush=True)
gpt2 = tiktoken.get_encoding("gpt2")
work = Path("/kaggle/working/ktok")
work.mkdir(parents=True, exist_ok=True)


def decode_shard(path: Path, n_tokens: int, out_path: Path) -> int:
    """Decode uint16 GPT-2 tokens to text. 50256 -> <|endoftext|>."""
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    n = min(n_tokens, len(arr))
    written = 0
    with open(out_path, "w") as fh:
        for start in range(0, n, 1_000_000):
            chunk = np.asarray(arr[start:min(start + 1_000_000, n)], dtype=np.int64)
            # tiktoken's decode handles the EOT id; keep it as the special token.
            text = gpt2.decode(chunk.tolist())
            fh.write(text)
            written += len(text)
            if start % 10_000_000 == 0:
                print(f"decoded {start:,}/{n:,} tokens", flush=True)
    return written


corpus_path = work / "corpus.txt"
val_path = work / "val.txt"
train_bytes = decode_shard(train_bin, TRAIN_TOKENS, corpus_path)
val_bytes = decode_shard(val_bin, VAL_TOKENS, val_path)
print("corpus bytes:", train_bytes, "val bytes:", val_bytes, flush=True)


def build_tokenizer(vocab: int, digits: bool = False) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    if digits:
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ])
    else:
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )
    tok.train([str(corpus_path)], trainer)
    return tok


def measure(tok: Tokenizer | None, text: str, label: str) -> dict:
    if tok is None:
        ids = gpt2.encode_ordinary(text)
        tokens = len(ids)
        decoded = gpt2.decode(ids)
    else:
        enc = tok.encode(text)
        tokens = len(enc.ids)
        decoded = tok.decode(enc.ids)
    nbytes = len(text.encode("utf-8"))
    words = len(text.split())
    return {
        "label": label,
        "bytes": nbytes,
        "tokens": tokens,
        "bytes_per_token": nbytes / max(1, tokens),
        "tokens_per_word": tokens / max(1, words),
        "uniform_bpb": (np.log2(50000 if tok is None else tok.get_vocab_size())
                        / max(1e-9, nbytes / max(1, tokens))),
        "roundtrip": decoded == text,
    }


val_text = val_path.read_text()[:2_000_000]
code_hits = sorted(Path("/kaggle/input").rglob("model.py"))
code_text = code_hits[0].read_text() if code_hits else "def f(x):\n    return x + 1\n"
number_text = " ".join(str(i) for i in range(20000)) + "\n" + "\n".join(
    f"{a} + {b} = {a + b}" for a in range(1, 2001) for b in (7, 13)
)
samples = {
    "heldout_english": val_text,
    "code": code_text,
    "numbers": number_text,
}

report = {
    "experiment": "K-Tok-v0",
    "train_bin": str(train_bin),
    "val_bin": str(val_bin),
    "train_tokens": TRAIN_TOKENS,
    "val_tokens": VAL_TOKENS,
    "train_bytes": train_bytes,
    "val_bytes": val_bytes,
    "tokenizers": {},
}
hashes = {}

for vocab in VOCABS:
    name = f"ktok_{vocab // 1000}k"
    print(f"\n=== training {name} ===", flush=True)
    tok = build_tokenizer(vocab)
    path = work / f"{name}.json"
    tok.save(str(path))
    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["tokenizers"][name] = {
        "vocab_size": tok.get_vocab_size(),
        "bytes_per_token": {},
        "uniform_bpb": {},
        "roundtrip": {},
        "samples": {},
    }
    for sample_name, text in samples.items():
        m = measure(tok, text, sample_name)
        report["tokenizers"][name]["samples"][sample_name] = m
    print(f"{name} vocab {tok.get_vocab_size()} saved {path}", flush=True)

# digit-aware 32k variant
try:
    print("\n=== training ktok_32k_digits ===", flush=True)
    tok_d = build_tokenizer(32000, digits=True)
    path = work / "ktok_32k_digits.json"
    tok_d.save(str(path))
    hashes["ktok_32k_digits"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["tokenizers"]["ktok_32k_digits"] = {
        "vocab_size": tok_d.get_vocab_size(),
        "samples": {k: measure(tok_d, v, k) for k, v in samples.items()},
    }
except Exception as exc:  # noqa: BLE001
    report["digit_variant_error"] = repr(exc)
    print("digit variant failed:", repr(exc), flush=True)

# GPT-2 baseline
report["tokenizers"]["gpt2"] = {
    "vocab_size": 50257,
    "samples": {k: measure(None, v, k) for k, v in samples.items()},
}
report["hashes"] = hashes
report["param_budget"] = {
    name: {
        "vocab_size": v["vocab_size"],
        "tied_embedding_params": v["vocab_size"] * 512,
        "savings_vs_gpt2": (50257 - v["vocab_size"]) * 512,
    }
    for name, v in report["tokenizers"].items()
}
(work / "ktok_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)

print("\nK-TOK v0")
for name, v in report["tokenizers"].items():
    eng = v["samples"]["heldout_english"]
    code = v["samples"]["code"]
    num = v["samples"]["numbers"]
    print(f"  {name:16s} vocab {v['vocab_size']:6d} "
          f"eng {eng['bytes_per_token']:.3f} B/tok "
          f"code {code['bytes_per_token']:.3f} "
          f"num {num['bytes_per_token']:.3f} "
          f"roundtrip {eng['roundtrip']}")
print("  report: /kaggle/working/ktok/ktok_report.json")
