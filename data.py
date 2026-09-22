"""Memory-mapped uint16 token shards for KALIA training."""

from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    """Random contiguous windows from a flat uint16 token file."""

    def __init__(self, path: str | Path, context_len: int):
        self.path = Path(path)
        self.context_len = context_len
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.tokens) < context_len + 2:
            raise ValueError(
                f"{self.path} has too few tokens ({len(self.tokens)}) for context_len={context_len}"
            )

    def __len__(self) -> int:
        return len(self.tokens)

    def get_batch(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
