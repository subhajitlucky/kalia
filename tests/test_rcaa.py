"""X27 is withdrawn: the reverse stream, as designed, leaked future tokens.

The first implementation's causality test failed -- the model's output at
position 11 changed when token 12 changed -- because the reverse stream at
position p attends to original positions q >= p. A next-token model that can
read the token it is being asked to predict would post a spectacular
validation loss for entirely the wrong reason.

The config flag is kept as a tripwire: enabling it raises with the reason, so
the lesson lives exactly where someone would try to switch it on.
See docs/preregistrations/2026-10-09-X27-amendment-1.md.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import GPT, GPTConfig  # noqa: E402


def test_the_withdrawn_arm_cannot_be_enabled_silently():
    cfg = GPTConfig(
        vocab_size=64, n_layer=2, n_head=2, n_embd=32, context_len=16,
        reverse_attention=True,
    )
    with pytest.raises(ValueError, match="X27-am1"):
        GPT(cfg)


def test_the_flag_defaults_off():
    m = GPT(GPTConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=32, context_len=16))
    assert m.cfg.reverse_attention is False
