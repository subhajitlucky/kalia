"""Memory-mapped uint16 token shards for KALIA training."""

from pathlib import Path

import numpy as np
import torch

# GPT-2's end-of-text token. Measured on our own corpus: all four sources
# (FineWeb-Edu, TinyStories, Cosmopedia, code) are delimited by this same token,
# at a comparable rate (2.1-5.0 per 1k tokens, median document ~200 tokens).
# A single global delimiter is therefore sufficient for document masking.
DOC_SEPARATOR = 50256


def document_mask(
    ids: torch.Tensor, separator: int = DOC_SEPARATOR
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-diagonal document mask for a batch of token windows.

    Returns ``(attn_mask, loss_mask)`` for a window ``ids`` of shape (B, T).

    ``doc_id[i]`` counts separators strictly before position ``i``, so all
    positions sharing a doc id belong to the same document. Attention is
    allowed only within a document *and* causally (position i sees j <= i).
    The loss is taken only where the target stays inside the same document,
    i.e. where the predicted next token is not the first token of a new one.

    Documents are short (median ~200 tokens) relative to our 1024-token
    context, so most windows previously spanned several documents and the
    model attended across every boundary. See docs/journal for the measurement.
    """
    b, t = ids.shape
    is_sep = ids == separator
    # doc_id[i] = number of separators in ids[:, :i]
    doc_id = torch.cumsum(is_sep.long(), dim=1) - is_sep.long()
    same_doc = doc_id.unsqueeze(2) == doc_id.unsqueeze(1)  # (B, T, T): i attends j
    causal = torch.ones(t, t, dtype=torch.bool, device=ids.device).tril()
    attn = (same_doc & causal).unsqueeze(1)  # (B, 1, T, T)
    # Target y[i] is x[i+1]; it opens a new document exactly when x[i] is a
    # separator. Full (B, T) shape, matching the whole target window.
    loss_mask = ~is_sep
    return attn, loss_mask


class TokenDataset:
    """Random contiguous windows from a flat uint16 token file.

    Optional ``reversal_prob`` applies chunk-preserving segment reversal to a
    fraction of training windows (X16, Chakravyuha): the window is split into
    random chunks of ``reversal_min_chunk``..``reversal_max_chunk`` tokens and
    the chunk order is reversed, while token order inside each chunk is kept.
    The same position permutation is applied to x and y, so every target stays
    the true successor of its input token in the original stream.
    """

    def __init__(
        self,
        path: str | Path,
        context_len: int,
        reversal_prob: float = 0.0,
        reversal_min_chunk: int = 4,
        reversal_max_chunk: int = 16,
    ):
        self.path = Path(path)
        self.context_len = context_len
        self.reversal_prob = reversal_prob
        self.reversal_min_chunk = reversal_min_chunk
        self.reversal_max_chunk = reversal_max_chunk
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.tokens) < context_len + 2:
            raise ValueError(
                f"{self.path} has too few tokens ({len(self.tokens)}) for context_len={context_len}"
            )

    def __len__(self) -> int:
        return len(self.tokens)

    def _segment_permutation(self, length: int, generator: torch.Generator) -> torch.Tensor:
        chunks = []
        pos = 0
        while pos < length:
            size = int(
                torch.randint(
                    self.reversal_min_chunk,
                    self.reversal_max_chunk + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            size = min(size, length - pos)
            chunks.append((pos, pos + size))
            pos += size
        perm = [i for start, end in reversed(chunks) for i in range(start, end)]
        return torch.tensor(perm, dtype=torch.long)

    def _apply_reversal(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for row in range(x.shape[0]):
            if float(torch.rand(1, generator=generator).item()) < self.reversal_prob:
                perm = self._segment_permutation(x.shape[1], generator)
                x[row] = x[row][perm]
                y[row] = y[row][perm]
        return x, y

    def get_batch(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        return_masks: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        max_start = len(self.tokens) - self.context_len - 1
        starts = torch.randint(max_start, (batch_size,), generator=generator)
        x = torch.stack(
            [torch.from_numpy(self.tokens[i : i + self.context_len].astype(np.int64)) for i in starts]
        )
        y = torch.stack(
            [
                torch.from_numpy(self.tokens[i + 1 : i + 1 + self.context_len].astype(np.int64))
                for i in starts
            ]
        )
        if self.reversal_prob > 0:
            x, y = self._apply_reversal(x, y, generator)
        if not return_masks:
            return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        attn_mask, loss_mask = document_mask(x)
        return (
            x.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
            attn_mask.to(device, non_blocking=True),
            loss_mask.to(device, non_blocking=True),
        )
