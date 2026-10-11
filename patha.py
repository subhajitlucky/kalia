"""Patha replay: Vedic recitation orderings as replay schedules (X29).

The pāṭha methods are structured rehearsal algorithms from the Vedic oral
tradition — a corpus preserved for three thousand years by memory alone:

    krama:  overlapping forward pairs          12, 23, 34, ...
    jata:   forward-reverse-forward            12, 21, 12, 23, 32, 23, ...
    ghana:  nested triples                     12, 21, 123, 321, 123, ...

Registered in docs/preregistrations/2026-10-11-X29-patha-replay.md. The 2026
replay-scheduling field (SRT/SuperMemo-2, FOREVER, Replay-on-Demand, MSSR) all
schedules from psychology; this is the first test of recitation-derived
orderings as a training schedule.

The fairness constraint is structural: every arm draws from the *same* base
window set at the *same* replay budget; only the order and repetition pattern
within the replay stream differ. Orderings repeat examples (jata recites each
pair three times), so the arms are equalised by budget (10% of rows), not by
cycle length — the pattern cycles, the rate stays fixed.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from data import TokenDataset, document_mask

PATTERNS = ("uniform", "krama", "jata", "ghana")


def patha_stream(m: int, pattern: str, seed: int = 0) -> list[int]:
    """One cycle of base-window indices (0..m-1) in the pattern's order.

    ``uniform`` is a seeded shuffle — the control: same window set, no
    structure. The recitation patterns are transcribed from the pāṭha
    descriptions: krama links overlapping pairs forward; jata recites each
    pair forward-reverse-forward; ghana nests triples.
    """
    if m < 1:
        raise ValueError("need at least one base window")
    if pattern == "uniform":
        order = list(range(m))
        random.Random(seed).shuffle(order)
        return order
    if pattern == "krama":
        return [j for i in range(m - 1) for j in (i, i + 1)]
    if pattern in ("jata", "ghana"):
        need = 2 if pattern == "jata" else 3
        if m < need:
            raise ValueError(f"{pattern} needs at least {need} base windows, got {m}")
        out: list[int] = []
        if pattern == "jata":
            for i in range(m - 1):
                a, b = i, i + 1
                out += [a, b, b, a, a, b]
        else:
            for i in range(m - 2):
                a, b, c = i, i + 1, i + 2
                out += [a, b, b, a, a, b, c, c, b, a, a, b, c]
        return out
    raise ValueError(f"unknown pattern {pattern!r}; expected one of {PATTERNS}")


class PathaReplayDataset:
    """MixtureDataset's interface, with replay rows ordered by a pāṭha pattern.

    Same contract as :class:`data.MixtureDataset` (``context_len``,
    ``__len__``, ``get_batch``) so ``train.py`` works unchanged. Replay rows
    come from a fixed base-window set walked in the pattern's order; new-data
    rows are drawn exactly as before. The replay *rate* is the registered
    budget; the pattern changes only the presentation order.
    """

    def __init__(
        self,
        new_data: TokenDataset,
        replay_data: TokenDataset,
        replay_prob: float = 0.10,
        pattern: str = "uniform",
        base_windows: int = 256,
        seed: int = 0,
    ):
        if not 0.0 <= replay_prob < 1.0:
            raise ValueError(f"replay_prob must be in [0, 1), got {replay_prob}")
        if pattern not in PATTERNS:
            raise ValueError(f"unknown pattern {pattern!r}; expected one of {PATTERNS}")
        self.new_data = new_data
        self.replay_data = replay_data
        self.replay_prob = float(replay_prob)
        self.pattern = pattern
        max_start = len(replay_data.tokens) - replay_data.context_len - 1
        if max_start < 1:
            raise ValueError("replay shard too short for one window")
        rng = random.Random(seed)
        n_windows = min(base_windows, max_start)
        self._base_offsets = sorted(rng.sample(range(max_start), n_windows))
        self._stream = patha_stream(n_windows, pattern, seed)
        self._cursor = 0
        # Own generator for the replay-count draws, so a validation pass that
        # passes no generator cannot consume the training RNG (same reason
        # MixtureDataset keeps one).
        self._rng = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return len(self.new_data)

    @property
    def context_len(self) -> int:
        return self.new_data.context_len

    def _replay_rows(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.replay_data.context_len
        tokens = self.replay_data.tokens
        xs, ys = [], []
        for _ in range(n):
            index = self._stream[self._cursor % len(self._stream)]
            self._cursor += 1
            start = self._base_offsets[index]
            xs.append(torch.from_numpy(tokens[start : start + ctx].astype(np.int64)))
            ys.append(torch.from_numpy(tokens[start + 1 : start + 1 + ctx].astype(np.int64)))
        return torch.stack(xs), torch.stack(ys)

    def get_batch(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        return_masks: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        gen = generator if generator is not None else self._rng
        n_replay = int(
            torch.binomial(
                torch.tensor([float(batch_size)]),
                torch.tensor([self.replay_prob]),
                generator=gen,
            ).item()
        )
        parts = []
        if n_replay > 0:
            parts.append(self._replay_rows(n_replay))
        if batch_size - n_replay > 0:
            parts.append(self.new_data.get_batch(batch_size - n_replay, device, gen))
        x = torch.cat([p[0] for p in parts], dim=0)
        y = torch.cat([p[1] for p in parts], dim=0)
        if not return_masks:
            return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        attn_mask, loss_mask = document_mask(x)
        return (
            x.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
            attn_mask.to(device, non_blocking=True),
            loss_mask.to(device, non_blocking=True),
        )
