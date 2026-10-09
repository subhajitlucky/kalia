"""Per-source sampling and per-source loss attribution, for the Kautilya arm.

Why this file exists
--------------------
Step 4 of the v0.3.0 pre-registration tests whether online mixture reweighting
(Kautilya's "four strategies": keep, upsample, downsample, freeze) beats the
static 60/20/15/5 mixture. It has a known implementation prerequisite, recorded
in the pre-registration before any of it ran:

    the current data path pre-blends shards offline in mix_bins.py into a single
    .bin, and data.py memory-maps one file. Runtime per-source weighting does not
    exist yet.

This is that missing piece. Two capabilities, because a reweighting experiment
without per-source measurement is just a mixture change nobody can interpret:

1. ``SourceMixtureDataset`` samples per source at runtime weights, replacing the
   pre-blended file.
2. ``per_source_loss`` attributes held-out loss to a source, which is the signal
   the Kautilya rule consumes and the secondary metric the pre-registration
   registered ("if adaptive and static tie, the tie-break is whether per-source
   losses are more balanced").

Design constraints
------------------
- **Same interface as ``TokenDataset``** (``context_len``, ``__len__``,
  ``get_batch``) so ``train.py`` works unchanged. Any caller that does
  ``isinstance(ds, MixtureDataset)`` for replay is unaffected, and this class
  deliberately does *not* subclass ``MixtureDataset`` to avoid inheriting replay
  semantics it does not implement.
- **Weights are set on the sampler, not baked in.** A run that re-blends on disk
  and a run that reweights at runtime are different experiments, and conflating
  them is how a mixture result becomes uninterpretable.
- **Determinism.** One ``torch.Generator`` drives source choice and offset choice,
  so a given seed reproduces the exact same per-source counts. The Kautilya arm
  compares adaptive against static at equal tokens; if sampling were not
  reproducible the arms would differ in more than the policy.

Usage
-----
    ds = SourceMixtureDataset(
        sources={"fineweb": fw_ds, "tinystories": ts_ds, ...},
        weights={"fineweb": 0.60, "tinystories": 0.20, ...},
    )
    x, y, src_ids = ds.get_batch(64, device, generator=g, return_source=True)
    losses = per_source_loss(model, ds, device, generator=g)  # {"fineweb": 2.9, ...}

Kautilya mapping (``update_weights_from_signal``)
-------------------------------------------------
The four strategies are expressed against a source's recent loss trend:

    improving  -> dana     (upsample)   : being learned, feed it more
    flat       -> sama     (keep)       : neutral, hold the weight
    worsening  -> bheda    (downsample) : hurting, reduce it
    collapsed  -> danda    (freeze)     : diverging, zero it until it recovers

The trend is measured on a source's own held-out loss, never on its training
loss. Training loss is confounded by how much of that source the model has just
seen, so a source that was heavily sampled looks worse purely for being
sampled -- which would invert the policy's own intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from data import TokenDataset, document_mask

# Kautilya's four strategies.
SAMA, DANA, BHEDA, DANDA = "sama", "dana", "bheda", "danda"


@dataclass
class SourceMixtureDataset:
    """Samples context windows across named sources at runtime-set weights.

    Parameters
    ----------
    sources:
        Non-empty mapping of source name -> ``TokenDataset``. Each is memory-mapped
        already; this class only chooses which one supplies each row.
    weights:
        Source name -> sampling weight. Need not be normalised and need not sum to
        1; they are normalised internally. Must cover exactly the keys of
        ``sources``, because a silently ignored source is a silent change to the
        experiment.
    min_weight:
        Floor applied after normalisation so a source cannot be driven to exactly
        zero by a transient loss spike. Exceeding the floor is what ``danda``
        (freeze) is for, and it is an explicit, auditable action rather than a
        consequence of arithmetic.
    """

    sources: dict[str, TokenDataset]
    weights: dict[str, float]
    min_weight: float = 0.01
    max_weight: float = 0.80
    reweight_momentum: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("need at least one source")
        if set(self.weights) != set(self.sources):
            raise ValueError(
                f"weights {sorted(self.weights)} do not match sources {sorted(self.sources)}"
            )
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("weights must be non-negative")
        lengths = {name: len(ds) for name, ds in self.sources.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"sources have different lengths {lengths}; equal-length shards keep "
                "source choice and token count from becoming entangled"
            )
        ctx = {ds.context_len for ds in self.sources.values()}
        if len(ctx) != 1:
            raise ValueError(f"sources disagree on context_len {ctx}")
        # Stable order so the source-id encoding is reproducible across runs.
        self._names: list[str] = sorted(self.sources)
        self._index = {name: i for i, name in enumerate(self._names)}
        self._probs = self._normalise(self.weights)
        # Own generator for callers that pass none. train.py's evaluate() calls
        # get_batch(batch, device) with no generator, which would otherwise draw
        # source choices from the *global* RNG -- making a validation pass depend
        # on how many batches training happened to draw first, and silently
        # breaking run-to-run reproducibility of the val number.
        self._rng = torch.Generator().manual_seed(int(self.seed))
        # Per-update loss history, for the trend-based policy (trend_window >= 2).
        # Bounded so a long run cannot grow memory without limit; 32 windows is
        # far more than any registered trend window needs.
        self._loss_history: list[dict[str, float]] = []

    def _normalise(self, w: Mapping[str, float]) -> torch.Tensor:
        v = torch.tensor([float(w[n]) for n in self._names], dtype=torch.float64)
        if float(v.sum()) <= 0:
            raise ValueError("weights sum to zero")
        if self.min_weight * len(self._names) > 1.0 + 1e-9 or self.max_weight * len(self._names) < 1.0 - 1e-9:
            raise ValueError(
                f"infeasible bounds: min_weight={self.min_weight}, "
                f"max_weight={self.max_weight} over {len(self._names)} sources"
            )
        if self.min_weight > 0 and self.max_weight < 1:
            v = self._project_bounded(v / float(v.sum()))
            if (
                float(v.max()) > self.max_weight + 1e-6
                or float(v.min()) < self.min_weight - 1e-6
                or abs(float(v.sum()) - 1.0) > 1e-6
            ):
                # The projection is a numerical routine; if it fails to land inside
                # the bounds it must say so rather than hand back a silent violation.
                raise ValueError(
                    "bounded projection failed to satisfy "
                    f"min={self.min_weight} max={self.max_weight} sum=1"
                )
        else:
            v = v / v.sum()
        return v.float()

    def _project_bounded(self, v: torch.Tensor) -> torch.Tensor:
        """Euclidean projection onto {sum = 1, lo <= v <= hi}.

        Clamp-then-renormalise is not a projection: dividing by a sum below one
        pushes every entry back up, so the ceiling is violated on the very next
        step. Two measured failures before this version existed -- clamp(max=0.8)
        then normalise gave 0.87, and iterating it drove a source to the floor
        while the mass could no longer sum to one (0.8 + 3*0.01 = 0.83).

        The correct construction is a single uniform shift found by bisection:
        v_i <- clip(v_i - theta, lo, hi) with theta chosen so the sum is one.
        The sum is non-increasing in theta, so bisection converges.
        """
        lo, hi = float(self.min_weight), float(self.max_weight)
        n = v.numel()

        def shifted(theta: float) -> torch.Tensor:
            return torch.clamp(v - theta, min=lo, max=hi)

        low, high = float(v.min()) - 1.0, float(v.max())
        for _ in range(200):
            mid = (low + high) / 2.0
            if float(shifted(mid).sum()) > 1.0:
                low = mid
            else:
                high = mid
        out = shifted((low + high) / 2.0)
        # Remove float drift from the sum by nudging entries that have slack.
        drift = 1.0 - float(out.sum())
        if abs(drift) > 1e-12:
            for i in range(n):
                room = (hi - out[i]) if drift > 0 else (out[i] - lo)
                step = drift if abs(drift) <= room else room
                out[i] = out[i] + step
                drift -= step
                if abs(drift) < 1e-12:
                    break
        return out

    # -- introspection -----------------------------------------------------

    @property
    def context_len(self) -> int:
        return next(iter(self.sources.values())).context_len

    @property
    def source_names(self) -> list[str]:
        return list(self._names)

    def probabilities(self) -> dict[str, float]:
        return {n: float(p) for n, p in zip(self._names, self._probs)}

    def set_probabilities(self, probs) -> None:
        """Replace sampling probabilities, always storing them on CPU.

        The DDP reweight path hands over a CUDA tensor (NCCL collectives
        require one), but this sampler draws with a CPU generator, and a
        CUDA-stored ``_probs`` breaks the next batch on the receiving rank.
        Found in the first real Kautilya run (X26, 2026-10-09): rank 1 was the
        only rank that received the tensor, and the run deadlocked in the
        following step's collectives.
        """
        values = torch.as_tensor(probs, dtype=torch.float32).detach().cpu()
        if values.numel() != len(self._names):
            raise ValueError(
                f"expected {len(self._names)} probabilities, got {values.numel()}"
            )
        if float(values.sum()) <= 0:
            raise ValueError("probabilities must sum to a positive value")
        self._probs = values

    def __len__(self) -> int:
        return len(next(iter(self.sources.values())))

    # -- sampling ----------------------------------------------------------

    def get_batch(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        return_masks: bool = False,
        return_source: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """One row per sample, each drawn from an independently chosen source.

        Source choice is per row rather than per batch. Per-batch would make the
        realised mixture a coarse quantisation of the requested weights, which at
        batch 64 is a 1.6% granularity -- large enough to confound a policy that
        moves weights by a few points.
        """
        gen = generator if generator is not None else self._rng
        picks = torch.multinomial(self._probs.expand(batch_size, -1), 1, generator=gen)
        picks = picks.squeeze(1)
        xs, ys = [], []
        for i in range(batch_size):
            ds = self.sources[self._names[int(picks[i])]]
            x, y = ds.get_batch(1, torch.device("cpu"), generator=gen)
            xs.append(x[0])
            ys.append(y[0])
        x = torch.stack(xs)
        y = torch.stack(ys)
        if not return_masks and not return_source:
            return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out: list[torch.Tensor] = [x.to(device, non_blocking=True), y.to(device, non_blocking=True)]
        if return_masks:
            attn_mask, loss_mask = document_mask(x)
            out += [attn_mask.to(device), loss_mask.to(device)]
        if return_source:
            out.append(picks.to(device))
        return tuple(out)

    # -- the Kautilya policy ------------------------------------------------

    def update_weights_from_signal(
        self,
        per_source_loss: Mapping[str, float],
        step_up: float = 1.10,
        step_down: float = 0.90,
        trend_window: int = 1,
        trend_eps: float = 0.01,
        collapse_margin: float = 0.5,
    ) -> dict[str, str]:
        """Reweight from each source's loss trend. Returns the strategy applied.

        ``trend_window == 1`` keeps the level-vs-mean rule the first pilot ran
        under (X26): a source below the mean is upsampled whether or not it is
        still improving, which its diary showed drifting toward "eating
        dessert" -- feeding what is already easy. That rule is retained
        unchanged so the pilot's recorded behaviour stays reproducible.

        ``trend_window >= 2`` implements the A1 design faithfully: the newest
        window's mean is compared against the preceding window's mean, relative
        to ``trend_eps`` (fraction of the previous mean):

            improving  -> dana   (upsample)
            flat       -> sama   (keep)
            worsening  -> bheda  (downsample)
            collapsed  -> danda  (freeze at the floor), when a source is
                          ``collapse_margin`` nats above its best-ever mean

        Until ``2 * trend_window`` evaluations exist the policy falls back to
        the level rule, because a trend needs two windows to exist at all.

        The four outcomes follow the pre-registration exactly. Improvement
        always wins, because a source the model is still learning from is the
        one worth feeding.
        """
        if not per_source_loss:
            raise ValueError("no per-source loss supplied")
        unknown = set(per_source_loss) - set(self._names)
        if unknown:
            raise ValueError(f"loss supplied for unknown sources {sorted(unknown)}")
        if trend_window < 1:
            raise ValueError(f"trend_window must be >= 1, got {trend_window}")

        losses = {n: float(per_source_loss[n]) for n in self._names}
        self._loss_history.append(dict(losses))
        del self._loss_history[:-32]

        w = dict(self.weights)
        applied: dict[str, str] = {}
        # Momentum damps the step when the signal is noisy, matching the intent of
        # trend_window>1: a single evaluation should not be able to move the whole
        # mixture. 0.0 disables it and reproduces the plain multiplicative rule.
        gamma = self.reweight_momentum

        use_trend = trend_window >= 2 and len(self._loss_history) >= 2 * trend_window
        if use_trend:
            recent = self._loss_history[-trend_window:]
            previous = self._loss_history[-2 * trend_window : -trend_window]
            best = {n: min(window[n] for window in self._loss_history) for n in self._names}
            for name in self._names:
                new_mean = sum(window[name] for window in recent) / trend_window
                prev_mean = sum(window[name] for window in previous) / trend_window
                eps = abs(prev_mean) * trend_eps
                if new_mean > best[name] + collapse_margin:
                    strategy = DANDA
                    w[name] = self.min_weight
                elif new_mean < prev_mean - eps:
                    strategy = DANA
                    w[name] = w[name] * step_up
                elif new_mean > prev_mean + eps:
                    strategy = BHEDA
                    w[name] = w[name] * step_down
                else:
                    strategy = SAMA
                if gamma > 0:
                    w[name] = w[name] ** (1.0 - gamma)
                applied[name] = strategy
        else:
            mean = sum(losses.values()) / len(losses)
            for name in self._names:
                loss = losses[name]
                # Strictly better, not "at or better". With `<=` a source sitting
                # exactly at the mean is labelled dana, so a set of identical losses
                # labels every source as "improving" -- which is a policy that reports
                # progress while having none, and normalises to a no-op. At the mean is
                # precisely the sama case.
                better_than_mean = loss < mean - 1e-9
                if better_than_mean:
                    strategy = DANA
                    w[name] = w[name] * step_up
                elif loss <= mean * 1.15:
                    strategy = SAMA
                else:
                    strategy = BHEDA
                    w[name] = w[name] * step_down
                if gamma > 0:
                    w[name] = w[name] ** (1.0 - gamma)
                applied[name] = strategy

        total = sum(w.values())
        if total <= 0:
            return applied
        # Renormalise, then clip through the same floor/ceiling the constructor
        # uses, so the sampler can never hold a weight the bounds forbid.
        self._probs = self._normalise(w)
        self.weights = {n: float(p) for n, p in zip(self._names, self._probs)}
        return applied


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when wrapped by DDP."""
    return getattr(model, "module", model)


@torch.no_grad()
def per_source_loss(
    model: torch.nn.Module,
    mixture: SourceMixtureDataset,
    device: torch.device,
    generator: torch.Generator | None = None,
    batches: int = 4,
    batch_size: int = 16,
    micro_batch: int = 2,
) -> dict[str, float]:
    """Mean held-out loss per source, each source measured on its own data.

    Not the training loss. A source sampled heavily shows a worse *training* loss
    for the uninteresting reason that the model has just seen it, so a policy
    driven by training loss would downsample exactly the sources it just
    oversampled -- the opposite of the intended behaviour.

    Each source is evaluated on its own shard only, so the numbers are comparable
    across sources even though absolute loss differs by domain.

    The forward is chunked into ``micro_batch`` rows because the full-batch
    logits are ~3 GiB at batch 16 x context 1024 (16,384 x 50,257 fp32) and the
    training process already holds most of the card. The first real run of the
    Kautilya policy died with CUDA OOM here (X26, 2026-10-09); chunking is a
    memory fix, not a protocol change -- the mean is over the same tokens, and a
    test asserts the chunked and full-batch numbers are identical.
    """
    if batches < 1:
        raise ValueError("batches must be >= 1")
    if micro_batch < 1:
        raise ValueError("micro_batch must be >= 1")
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for name in mixture.source_names:
            ds = mixture.sources[name]
            total_loss, total_tokens = 0.0, 0
            for _ in range(batches):
                x, y = ds.get_batch(batch_size, device, generator=generator)
                attn_mask, loss_mask = document_mask(x.cpu())
                for start in range(0, batch_size, micro_batch):
                    xb = x[start : start + micro_batch]
                    yb = y[start : start + micro_batch]
                    mb = attn_mask[start : start + micro_batch].to(device)
                    lm = loss_mask[start : start + micro_batch].to(device)
                    # KALIA's GPT.forward returns (logits, loss) and takes `targets`
                    # plus optional attn_mask/loss_mask. It does not accept a
                    # loss_mask keyword and does not return bare logits. Calling it
                    # the way a generic model wrapper would returns a tuple -- which
                    # is exactly what happened before this was fixed, and it only
                    # surfaced when the function met the real model inside train.py.
                    # Unwrap defensively so a DDP-wrapped model works too.
                    raw = _unwrap(model)
                    out = raw(xb, yb, attn_mask=mb, loss_mask=lm)
                    logits = out[0] if isinstance(out, tuple) else out
                    b, t, v = logits.shape
                    ce = F.cross_entropy(
                        logits.reshape(b * t, v),
                        yb.reshape(b * t),
                        reduction="none",
                    )
                    total_loss += float(ce.sum().item())
                    total_tokens += b * t
            totals[name] = total_loss / max(total_tokens, 1)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if was_training:
            model.train()
    return totals


def strategy_counts(applied: Mapping[str, str]) -> dict[str, int]:
    """How many sources got each strategy. Used to assert all four are reachable."""
    out = {SAMA: 0, DANA: 0, BHEDA: 0, DANDA: 0}
    for s in applied.values():
        out[s] = out.get(s, 0) + 1
    return out
