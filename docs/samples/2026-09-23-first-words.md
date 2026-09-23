# KALIA Samples — First Real Words

Date: 2026-09-23
Checkpoint: v0.1.2 (Muon + QK-Norm + soft-cap), `checkpoints/ckpt.pt`
Step: 1738 / 4770 (~911M tokens seen, 37% of planned training)
Validation loss: ~3.22 (baseline v0.1.0 final: 3.2702 at step 2250)

Generated locally on CPU with `sample.py` (temperature 0.8, top-k 200).

## Prompt: "Once upon a time"

Once upon a time, there was a little girl named Lily. Lily loved to play in the
park with her friends. One day, Lily was playing with her friends when her mom
called out, "Lily, it's time to go home. Please come out and play with me
again."

Lily felt sad because she really wanted to play with her friends. But then she
had an idea. "I

## Prompt: "The little fox"

The little fox was a little girl named Lily who loved to play. One day, she
found a long wire in the grass. She picked it up and noticed a little bug
crawling around. "Hello, little bug," she said to herself.

The little bug replied, "Hello, Lily. Can I play with you?"

"Sure, you can play with me," said Lily.

## Notes

- Grammar, punctuation, dialogue formatting and narrative structure are correct.
- TinyStories domain artifacts ("Lily") are expected; that is the training data.
- Occasional entity swaps ("fox" becomes "Lily") are typical at this scale and
  step count; more training narrows them.
