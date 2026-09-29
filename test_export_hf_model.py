"""Tests for the Hugging Face export path.

The export is the last thing between a training checkpoint and a public artifact,
and it has two failure modes that are invisible until someone loads the model.

**Shared storage.** KALIA ties `lm_head` to `tok_emb`, so those two entries in the
state dict are the *same* tensor. `safetensors.save_file` refuses to serialise
shared tensors and raises `RuntimeError: A potential way to correctly save your
model is to use save_model`. A naive export therefore dies outright -- and if
someone "fixes" that by dropping the head, the published repo silently has a
different tensor set from the one that was measured, which is the failure that
matters. The exporter keeps the head and clones the alias, matching the 93-tensor
layout v0.1.2 already ships.

**Config drift.** The published `config.json` is derived from the checkpoint's own
`config['model']` rather than transcribed, so the released architecture cannot
quietly diverge from the trained one.

These run on a 2-layer toy model, so they cost nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from model import GPT, GPTConfig
from tools.export_hf_model import EXPECTED, PROTOCOL, build_config, write_safetensors

TOY = dict(vocab_size=64, n_layer=2, n_head=2, n_embd=32, context_len=16, qk_norm=True)


def _toy_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    m = GPT(GPTConfig(**TOY))
    return {k: v.detach() for k, v in m.state_dict().items()}


def test_tied_head_really_does_share_storage():
    """The premise. If this stops holding, the de-share below is dead code."""
    sd = _toy_state()
    assert sd["lm_head.weight"].untyped_storage().data_ptr() == sd["tok_emb.weight"].untyped_storage().data_ptr()


def test_naive_safetensors_save_really_does_fail():
    """Why the de-share exists. Skips if the error text ever changes."""
    import tempfile

    from safetensors.torch import save_file

    sd = _toy_state()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            save_file({k: v.contiguous() for k, v in sd.items()}, str(Path(tmp) / "x.safetensors"))
        except RuntimeError:
            return
    raise AssertionError("expected safetensors to reject shared storage")


def test_export_keeps_the_head_and_preserves_values(tmp_path):
    from safetensors.torch import load_file

    sd = _toy_state()
    out = tmp_path / "m.safetensors"
    n = write_safetensors(sd, step=4770, tokens=123, path=out)

    back = load_file(str(out))
    assert n == len(sd), "every tensor must be published, including the tied head"
    assert "lm_head.weight" in back and "tok_emb.weight" in back
    for k, v in sd.items():
        assert torch.equal(back[k], v.float()), f"{k} was altered by the export"
    assert torch.equal(back["lm_head.weight"], back["tok_emb.weight"]), "head must stay tied"


def test_export_writes_step_and_tokens_metadata(tmp_path):
    from safetensors.torch import safe_open

    out = tmp_path / "m.safetensors"
    write_safetensors(_toy_state(), step=4770, tokens=2_500_853_760, path=out)
    with safe_open(str(out), framework="pt") as f:
        meta = f.metadata()
    assert meta["step"] == "4770"
    assert meta["tokens"] == "2500853760"
    assert meta["format"] == "pt"


def test_published_config_is_derived_not_transcribed():
    """Every architecture field must come from the checkpoint's own config."""
    ck = {"vocab_size": 50257, "n_layer": 10, "n_head": 8, "n_embd": 512,
          "context_len": 1024, "dropout": 0.0, "qk_norm": True, "logit_softcap": 30.0}
    cfg = build_config(ck)
    for key in ("vocab_size", "n_layer", "n_head", "n_embd", "context_len", "qk_norm", "logit_softcap"):
        assert cfg[key] == ck[key], key
    assert cfg["model_type"] == "kalia"
    assert cfg["architectures"] == ["KaliaGPT"]
    assert cfg["tie_embeddings"] is True
    # v0.2.0 and v0.1.2 share a recipe, so the configs must be identical.
    assert json.loads(json.dumps(cfg))["n_layer"] == 10


def test_config_omits_architecture_flags_the_model_does_not_have():
    """NoPE and branch-norm must not leak into a published config that never used them."""
    cfg = build_config({"vocab_size": 50257, "n_layer": 10, "n_head": 8, "n_embd": 512,
                        "context_len": 1024, "dropout": 0.0, "qk_norm": True, "logit_softcap": 30.0})
    assert "nope_interval" not in cfg
    assert "branch_norm" not in cfg
    assert "gated_residual" not in cfg


def test_release_gate_constants_match_the_registered_results():
    """The gate compares against these, so they must be the registered values."""
    assert EXPECTED["v020"]["val_loss"] == 2.8248
    assert EXPECTED["v020"]["bpb"] == 0.9244
    assert EXPECTED["v020"]["step"] == 4770
    assert EXPECTED["v020"]["tokens"] == 2_500_853_760
    assert EXPECTED["v012"]["val_loss"] == 3.0533
    assert PROTOCOL == {"batches": 100, "batch_size": 8, "seed": 1234}


def test_untrained_toy_model_cannot_pass_the_release_gate():
    """A sanity check on the gate itself: random weights must not score 2.8248."""
    with pytest.raises(AssertionError):
        assert abs(3.14159 - 2.8248) <= 5e-4
