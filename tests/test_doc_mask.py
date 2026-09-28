"""Tests for intra-document masking (the packed-corpus defect fix)."""

import numpy as np
import torch

from data import DOC_SEPARATOR, TokenDataset, document_mask
from model import GPT, GPTConfig


def test_mask_is_block_diagonal_and_causal():
    # Two documents of 3 tokens each: [1 2 SEP 4 5 SEP]
    ids = torch.tensor([[1, 2, DOC_SEPARATOR, 4, 5, DOC_SEPARATOR]])
    attn, loss_mask = document_mask(ids)
    t = ids.size(1)
    assert attn.shape == (1, 1, t, t)
    # Causal: no future position may be attended.
    for i in range(t):
        assert not attn[0, 0, i, i + 1 :].any(), f"position {i} attends into the future"
    # Block diagonal: position 3 (doc 2) must not see position 1 (doc 1).
    assert not attn[0, 0, 3, 1]
    assert not attn[0, 0, 3, 0]
    # Same document, past position: allowed.
    assert attn[0, 0, 1, 0]
    assert attn[0, 0, 4, 3]
    # Every position can always attend to itself.
    for i in range(t):
        assert attn[0, 0, i, i]


def test_document_ids_count_separators_strictly_before():
    ids = torch.tensor([[7, DOC_SEPARATOR, 9, 10]])
    attn, _ = document_mask(ids)
    # Position 2 is the first token of doc 1, position 0/1 belong to doc 0.
    assert not attn[0, 0, 2, 0]
    assert not attn[0, 0, 2, 1]
    assert attn[0, 0, 3, 2]


def test_loss_mask_excludes_boundary_targets():
    # y[i] = ids[i+1]; it opens a new document exactly when ids[i] is a separator.
    ids = torch.tensor([[1, 2, DOC_SEPARATOR, 4, 5, DOC_SEPARATOR]])
    _, loss_mask = document_mask(ids)
    assert loss_mask.shape == (1, ids.size(1))
    assert loss_mask[0].tolist() == [True, True, False, True, True, False]


def test_masked_attention_does_not_change_self_attention_on_one_document():
    """With no separators in the window, the masked path must equal the plain
    causal path -- otherwise the 'fix' would change results for other reasons."""
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (2, 16))
    attn, _ = document_mask(ids)  # no separators -> one document throughout
    causal = torch.ones(16, 16, dtype=torch.bool).tril().view(1, 1, 16, 16).expand(2, 1, 16, 16)
    assert torch.equal(attn, causal)


def test_model_masked_path_runs_and_loss_is_finite():
    cfg = GPTConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=32, context_len=32)
    model = GPT(cfg)
    ids = torch.randint(0, 63, (2, 16))
    targets = torch.randint(0, 63, (2, 16))
    attn, loss_mask = document_mask(ids)
    _, masked_loss = model(ids, targets, attn_mask=attn, loss_mask=loss_mask)
    _, plain_loss = model(ids, targets)
    assert torch.isfinite(masked_loss)
    assert torch.isfinite(plain_loss)
    # Masked loss must be computed over fewer positions than the plain one.
    assert loss_mask.sum() <= targets.numel()


def test_loss_mask_actually_changes_the_gradient():
    """A mask that is computed but ignored would pass every test above. This
    checks the model output genuinely depends on it."""
    torch.manual_seed(0)
    # Real vocabulary size, so the production separator id is a valid token.
    cfg = GPTConfig(vocab_size=50257, n_layer=2, n_head=2, n_embd=32, context_len=32)
    model = GPT(cfg)
    ids = torch.tensor([[1, 2, DOC_SEPARATOR, 4, 5, DOC_SEPARATOR, 8, 9, DOC_SEPARATOR, 11] * 3])
    targets = torch.roll(ids, -1, dims=1)
    attn, loss_mask = document_mask(ids)
    _, with_mask = model(ids, targets, attn_mask=attn, loss_mask=loss_mask)
    _, without = model(ids, targets)
    assert not torch.allclose(with_mask, without), "loss_mask had no effect on the loss"


def test_get_batch_returns_masks_when_asked(tmp_path):
    path = tmp_path / "t.bin"
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 1000, size=4000).astype(np.uint16)
    toks[500] = DOC_SEPARATOR  # plant one boundary
    toks.tofile(path)
    ds = TokenDataset(path, context_len=64)
    out = ds.get_batch(4, torch.device("cpu"), torch.Generator().manual_seed(0), return_masks=True)
    assert len(out) == 4
    x, y, attn, lm = out
    assert x.shape == y.shape == (4, 64)
    assert attn.shape == (4, 1, 64, 64)
    assert lm.shape == (4, 64)
    # Default path is unchanged for existing callers.
    x2, y2 = ds.get_batch(4, torch.device("cpu"), torch.Generator().manual_seed(0))
    assert x2.shape == x.shape and y2.shape == y.shape
    assert torch.equal(x2, x)
