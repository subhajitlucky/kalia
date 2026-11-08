# X56 — KALIA-Chat v0: null

Status: **null on C-2/C-3/C-4/C-7.** The adapter learned the chat format, the
refusal keyword and 2-turn copying, but not arithmetic, word manipulation,
context QA or held-out instructions. No v0.7.0-chat release; the adapter and
`chat.py` remain a research preview.

## Result

Frozen v0.3.1 + rank-32 LoRA (**2,826,240** trainable params), 2 epochs on
100k synthetic records, **40.5 minutes**. Held-out test (300/task, 2,100 total):

| task | accuracy | format |
|---|---:|---:|
| arithmetic | 0.247 | 1.000 |
| word | 0.067 | 0.987 |
| context QA | **0.000** | 1.000 |
| instruction | 0.350 | 1.000 |
| chit-chat | 0.000 | 1.000 |
| refusal | **1.000** | 1.000 |
| 2-turn memory | **0.717** | 1.000 |
| **overall** | **0.340** | **0.998** |

Frozen-base control: **0.000** overall (format 0.931). Best validation accuracy
during training: **0.594**; final training loss **0.16**. Checkpoint
`chat_lora.pt`, 11.3MB, sha `3191b281…`.

| bar | result |
|---|---|
| **C-1** format ≥ 0.95 | **PASS (0.998)** |
| **C-2** arithmetic ≥ 0.70 | **FAIL (0.247)** |
| **C-3** word tasks ≥ 0.70 | **FAIL (0.067)** |
| **C-4** context QA ≥ 0.80 | **FAIL (0.000)** |
| **C-5** refusal ≥ 0.90 | **PASS (1.000)** by keyword |
| **C-6** 2-turn memory ≥ 0.60 | **PASS (0.717)** |
| **C-7** instruction ≥ 0.60 | **FAIL (0.350)** |
| **C-8** base control | diagnostic: 0.000 |
| **C-9** from scratch; artifacts | **PASS** |

## Failure analysis

- **Context QA 0.000 is the clearest failure.** The model ignores the supplied
  fact and answers from memorized associations: “The sun is yellow” → `blue`;
  “A dog says woof” → `tweet`; “twenty-six letters” → `twenty`. It is not
  copying from the prompt.
- **Word 0.067:** character-level tasks fail. A memorized repeat
  (`music music music`) passes; first/last letter, uppercase and reversal do
  not.
- **Arithmetic 0.247:** some single- and two-digit sums work; there is no
  robust carry/borrow or subtraction.
- **Instruction 0.350:** templated sentences and color lists pass; numbers,
  alphabetical order and two-item lists fail.
- **Chit-chat 0.000:** the held-out “What is your name?” gets “I don’t know.”;
  the refusal prior dominates unknown questions.
- **Refusal 1.000 passes the registered keyword check, but the responses are
  degenerate** — “I can’t be kind.” for “Tell me a secret.” The check is too
  lenient; this is not a quality pass.
- **2-turn memory 0.717:** name and count copying work; chain arithmetic
  sometimes fails.

## Interpretation

The adapter learns the **surface format** and a few memorized patterns, but
not compositional manipulation, context copying or held-out instruction
composition. The 58M base + LoRA is capacity-limited for this suite, or the
synthetic data/objective is insufficient; X56 cannot separate the two. The
context-QA failure is the most important: a model that cannot copy a fact from
the prompt will not follow instructions reliably at this scale.

## Decision

No **v0.7.0-chat** release. The adapter and `chat.py` are a research preview:

```bash
python chat.py --adapter docs/results/x56-chat-v0/chat_lora.pt
```

Expect narrow, often incorrect answers. Public base remains **v0.3.1**. No
tuning after the result.

## Artifacts

- `metrics.json` — per-task accuracy, format, base control, samples.
- `predictions.jsonl` — 2,100 held-out predictions.
- `chat_lora.pt` — 11.3MB rank-32 adapter, sha `3191b281…`.
- `x56.log` — run log.
- Pre-registration: `docs/preregistrations/2026-11-08-X56-chat-v0.md`
  (hash `2eaa4be1…`).
- Kernel: `subhajitlucky/kalia-x56-chat`.
