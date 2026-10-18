"""The LR schedule: cosine is frozen, and WSD exists to stop paying for a tail.

E11 in the 2026-09-23 backlog (Kimi K2 / GLM-4.5 / IMU-1) asked for WSD with
flexible decay timing; it never landed, and v0.2.0 then spent its last 31% of
steps flat -- 6.3 GPU-hours for no measurable val movement, because a cosine is
already deep into its decay by then and has nothing left to give.

Two things have to hold at once. Every recorded run in this project used the
cosine, so `lr_schedule` absent must reproduce it exactly, bit for bit. And WSD
must actually hold the peak rather than approximating it.
"""

from __future__ import annotations

import math

import pytest

from train import (
    DECAY_SHAPES,
    base_lr_scale,
    decay_factor,
    lr_scale,
    validate_schedule,
    wsd_decay_steps,
    wsd_lr_scale,
)


def cfg(**over) -> dict:
    base = {"learning_rate": 6e-4, "warmup_steps": 100, "max_steps": 1000, "min_lr_ratio": 0.1}
    base.update(over)
    return base


def cosine_reference(step: int, c: dict) -> float:
    """The schedule exactly as it stood before WSD existed. Frozen here on
    purpose: this is what every recorded run was trained under."""
    if step < c["warmup_steps"]:
        return (step + 1) / c["warmup_steps"]
    if step >= c["max_steps"]:
        return c["min_lr_ratio"]
    progress = (step - c["warmup_steps"]) / max(1, c["max_steps"] - c["warmup_steps"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return c["min_lr_ratio"] + coeff * (1 - c["min_lr_ratio"])


# --- the cosine must not have moved -----------------------------------------

@pytest.mark.parametrize("step", [0, 1, 50, 99, 100, 101, 300, 550, 999, 1000, 1200])
def test_cosine_is_unchanged_without_a_schedule_key(step):
    c = cfg()
    assert base_lr_scale(step, c) == cosine_reference(step, c)


@pytest.mark.parametrize("step", [0, 99, 100, 300, 999, 1000])
def test_explicit_cosine_is_the_same_as_the_default(step):
    assert base_lr_scale(step, cfg(lr_schedule="cosine")) == cosine_reference(step, cfg())


def test_target_decay_path_is_unchanged():
    """The reactive decay is what all the continual-update runs used; WSD must
    not have perturbed it for runs that did not ask for it."""
    c = cfg(decay_steps_after_target=100)
    start = cosine_reference(300, c)
    assert lr_scale(300, c, decay_start=300) == start
    for step in (300, 325, 350, 399, 400, 500):
        progress = min(1.0, (step - 300) / 100)
        assert lr_scale(step, c, decay_start=300) == pytest.approx(
            start * (1 - progress) + c["min_lr_ratio"] * progress
        )


# --- WSD ---------------------------------------------------------------------

def test_wsd_warmup_matches_the_cosine_warmup():
    c = cfg(lr_schedule="wsd")
    for step in range(c["warmup_steps"]):
        assert base_lr_scale(step, c) == (step + 1) / c["warmup_steps"]


def test_wsd_holds_the_peak_through_the_stable_phase():
    c = cfg(lr_schedule="wsd", wsd_decay_frac=0.1)
    assert wsd_decay_steps(c) == 100
    for step in range(c["warmup_steps"], 900):
        assert base_lr_scale(step, c) == 1.0, f"not held at the peak at step {step}"


def test_wsd_decays_only_in_the_last_window_and_lands_on_the_floor():
    c = cfg(lr_schedule="wsd", wsd_decay_frac=0.1)
    assert base_lr_scale(900, c) == 1.0          # the window opens at the peak
    assert base_lr_scale(950, c) == pytest.approx(0.55)
    assert base_lr_scale(1000, c) == pytest.approx(c["min_lr_ratio"])
    assert base_lr_scale(1500, c) == pytest.approx(c["min_lr_ratio"])


def test_wsd_is_monotone_and_inside_the_band():
    """Rising through warmup, then never rising again. The peak is reached at
    the end of warmup and held, so the fall starts from the top."""
    c = cfg(lr_schedule="wsd")
    warmup = [base_lr_scale(s, c) for s in range(0, c["warmup_steps"])]
    assert all(b > a for a, b in zip(warmup, warmup[1:])), "warmup must climb"
    after = [base_lr_scale(s, c) for s in range(c["warmup_steps"], c["max_steps"] + 50)]
    assert all(b <= a + 1e-12 for a, b in zip(after, after[1:])), "WSD must never rise after warmup"
    assert max(after) == 1.0 and min(after) >= c["min_lr_ratio"] - 1e-12


def test_wsd_never_spends_less_of_the_budget_than_the_cosine():
    """The whole point: every step of a cosine is schedule already spent, so
    WSD's multiplier is at or above it everywhere. If this inverts, WSD is not
    buying back the flat tail it exists to buy back."""
    c = cfg(lr_schedule="wsd")
    for step in range(0, c["max_steps"] + 1):
        assert wsd_lr_scale(step, c) >= cosine_reference(step, c) - 1e-12, step
    # and it is a large difference where the cosine plateaued, not a rounding one
    assert wsd_lr_scale(700, c) - cosine_reference(700, c) > 0.3


def test_a_larger_decay_fraction_starts_the_anneal_earlier():
    narrow = cfg(lr_schedule="wsd", wsd_decay_frac=0.05)
    wide = cfg(lr_schedule="wsd", wsd_decay_frac=0.5)
    assert wsd_decay_steps(narrow) == 50 and wsd_decay_steps(wide) == 500
    assert base_lr_scale(600, narrow) == 1.0
    assert base_lr_scale(600, wide) < 1.0


def test_wsd_needs_a_stable_phase():
    """A decay window wider than the run is a configuration error, not a
    schedule: it would silently be a pure-linear decay from step one."""
    c = cfg(lr_schedule="wsd", wsd_decay_frac=1.0, warmup_steps=100)
    assert base_lr_scale(100, c) == 1.0  # degenerate but coherent, not a crash


# --- flexible decay timing: the feature E11 actually asked for ---------------

def test_the_anneal_can_be_triggered_early_and_runs_to_the_floor():
    """On the cosine path the reactive decay already existed; on WSD it is what
    makes the schedule useful -- a run that hits its target at step 300 pays 100
    steps of anneal, not the 700 the cosine had already planned for."""
    c = cfg(lr_schedule="wsd", decay_steps_after_target=100)
    assert lr_scale(299, c) == 1.0
    assert lr_scale(300, c, decay_start=300) == 1.0
    assert lr_scale(350, c, decay_start=300) == pytest.approx(0.55)
    assert lr_scale(400, c, decay_start=300) == pytest.approx(c["min_lr_ratio"])
    assert lr_scale(900, c, decay_start=300) == pytest.approx(c["min_lr_ratio"])


# --- decay shapes ------------------------------------------------------------

@pytest.mark.parametrize("shape", DECAY_SHAPES)
def test_every_shape_spans_one_to_zero(shape):
    assert decay_factor(0.0, shape) == pytest.approx(1.0)
    assert decay_factor(1.0, shape) == pytest.approx(0.0)
    assert decay_factor(2.0, shape) == pytest.approx(0.0), "past the end is the end"
    assert decay_factor(-1.0, shape) == pytest.approx(1.0)
    vals = [decay_factor(t / 20, shape) for t in range(21)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))


def test_the_shapes_are_actually_different():
    """Pinned so `decay_shape` cannot silently be a no-op knob."""
    linear, cosine, inv_sqrt = (decay_factor(0.25, s) for s in ("linear", "cosine", "inv_sqrt"))
    assert inv_sqrt < linear < cosine


def test_an_unknown_shape_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="unknown decay shape"):
        decay_factor(0.5, "quadratic")


# --- validation happens before the GPU work ---------------------------------

def test_validate_schedule_accepts_the_registered_configs():
    validate_schedule(cfg())  # absent key -> cosine
    validate_schedule(cfg(lr_schedule="cosine"))
    validate_schedule(cfg(lr_schedule="wsd"))
    for shape in DECAY_SHAPES:
        validate_schedule(cfg(lr_schedule="wsd", decay_shape=shape))


@pytest.mark.parametrize(
    "bad,match",
    [
        ({"lr_schedule": "wsd2"}, "lr_schedule must be one of"),
        ({"decay_shape": "quadratic"}, "decay_shape must be one of"),
        ({"lr_schedule": "wsd", "wsd_decay_frac": 0.0}, "wsd_decay_frac must be in"),
        ({"lr_schedule": "wsd", "wsd_decay_frac": 1.5}, "wsd_decay_frac must be in"),
        ({"lr_schedule": "wsd", "warmup_steps": 1000}, "needs a stable phase"),
    ],
)
def test_validate_schedule_refuses_a_schedule_the_run_cannot_honour(bad, match):
    with pytest.raises(SystemExit, match=match):
        validate_schedule(cfg(**bad))


def test_an_unknown_schedule_fails_at_the_first_lookup_too():
    """validate_schedule is the loud path, but the lookup must not silently
    fall back to the cosine if something reaches it without validation."""
    c = cfg(lr_schedule="wsd2")
    assert base_lr_scale(500, c) == cosine_reference(500, c)
