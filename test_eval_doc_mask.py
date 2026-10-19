"""The eval must mask attention when training did — and only then.

X17's +0.0444 penalty was an artifact of a train/eval mismatch the registration
had predicted in its own decision rule. These tests pin the two halves of the
fix: masked runs are now evaluated the way they trained, and unmasked runs are
bit-for-bit what every recorded result in this project used.

The second half is the one that matters most. Every published number — v0.1.2,
v0.2.0, all six architecture arms, every continual update — came from an
unmasked eval, and a "fix" that quietly moved them would invalidate the lot.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def _load_train():
    spec = importlib.util.spec_from_file_location("train_eval_under_test", ROOT / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = _load_train()


class FakeDataset:
    def __init__(self) -> None:
        self.return_masks_calls: list[bool] = []

    def get_batch(self, batch_size, device, generator=None, return_masks=False):
        self.return_masks_calls.append(return_masks)
        x = torch.zeros(2, 4, dtype=torch.long)
        y = torch.zeros(2, 4, dtype=torch.long)
        if return_masks:
            attn = torch.ones(2, 4, 4, dtype=torch.bool)
            loss_mask = torch.ones(2, 4, dtype=torch.bool)
            return x, y, attn, loss_mask
        return x, y


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[tuple[bool, bool]] = []

    def forward(self, x, y=None, attn_mask=None, loss_mask=None):
        self.calls.append((attn_mask is not None, loss_mask is not None))
        return None, torch.tensor(2.0)


def test_unmasked_eval_is_exactly_what_every_recorded_run_used():
    """No mask requested, no mask built, no mask passed."""
    ds, model = FakeDataset(), FakeModel()
    train.evaluate(model, ds, eval_steps=3)
    assert ds.return_masks_calls == [False, False, False]
    assert model.calls == [(False, False)] * 3


def test_masked_eval_masks_attention_and_leaves_the_loss_set_alone():
    """`attn_mask` is passed; `loss_mask` deliberately is not. Applying it would
    average the treatment over different tokens than the control, which is the
    incomparability this fix exists to remove."""
    ds, model = FakeDataset(), FakeModel()
    train.evaluate(model, ds, eval_steps=3, doc_mask=True)
    assert ds.return_masks_calls == [True, True, True]
    assert model.calls == [(True, False)] * 3, "attention masked, loss set untouched"


def test_the_default_is_unmasked():
    """A default that flipped would silently redefine every future number."""
    import inspect

    sig = inspect.signature(train.evaluate)
    assert sig.parameters["doc_mask"].default is False


def test_the_loss_returned_is_still_the_mean_over_steps():
    ds, model = FakeDataset(), FakeModel()
    mean, bpb = train.evaluate(model, ds, eval_steps=4)
    assert mean == 2.0
    assert bpb is None
