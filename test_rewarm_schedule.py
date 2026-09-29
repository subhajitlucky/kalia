"""The re-warm must actually re-warm, or Step 2 measures nothing.

Found while building the Step 2 notebook. Resuming the v0.2.0 checkpoint puts
`start_step` at 4770 of max_steps 4770, and base_lr_scale returns min_lr_ratio for
any step >= max_steps. So a resumed run sits at the decayed floor forever at
LR multiplier 0.100 -- and the Step 2 treatment arm, whose entire purpose is to
re-warm, would have been byte-identical to its control. It would have reported a
null for a mechanism it never ran, and the write-up would have said "the recipe
does not transfer at 58M" when the truth is "we never turned it on".

The weights resume; the clock does not.
"""

from __future__ import annotations

import math

import pytest
import yaml

CFG = yaml.safe_load(open("configs/kalia-v020.yaml"))["train"]
UPDATE = {**CFG, "max_steps": 500, "warmup_steps": 50}
ORIGIN = 4770  # where v0.2.0 ended


def base_lr_scale(step: int, c: dict) -> float:
    if step < c["warmup_steps"]:
        return (step + 1) / c["warmup_steps"]
    if step >= c["max_steps"]:
        return c["min_lr_ratio"]
    progress = (step - c["warmup_steps"]) / max(1, c["max_steps"] - c["warmup_steps"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return c["min_lr_ratio"] + coeff * (1 - c["min_lr_ratio"])


def test_unrebased_schedule_is_pinned_at_the_floor():
    """The bug, stated as a fact: this is what resume used to do."""
    vals = {base_lr_scale(ORIGIN + i, UPDATE) for i in range(0, 500, 10)}
    assert vals == {UPDATE["min_lr_ratio"]}, (
        "expected the pre-fix behaviour to be pinned at min_lr_ratio; if this "
        "changed, the test below is no longer proving what it claims"
    )


def test_rebased_schedule_reaches_peak_and_decays():
    steps = [base_lr_scale(i, UPDATE) for i in range(UPDATE["max_steps"] + 1)]
    peak = max(range(len(steps)), key=lambda i: steps[i])
    assert max(steps) > 0.9, "the re-warm must climb back toward the original peak"
    assert steps[0] < max(steps), "it must start low, not at the peak"
    # step 499 is not yet the floor; the floor is reached at max_steps. The point
    # is the trend, not landing exactly on the value one step early.
    assert steps[499] < steps[peak] * 0.5, "it must decay substantially by the end"
    assert steps[UPDATE["max_steps"]] == pytest.approx(UPDATE["min_lr_ratio"])
    # The shape: rise, then fall.
    assert 0 < peak < 499
    assert steps[peak] > steps[peak + 100]
    assert steps[peak] > steps[0]


def test_train_py_actually_rebases_the_schedule():
    """The test above models the schedule; this checks train.py calls it that way."""
    src = open("train.py").read()
    assert "lr_scale(step - lr_origin" in src, (
        "train.py must pass the update-local step to lr_scale, otherwise the "
        "rebasing is only in the test and not in the run"
    )
    assert "lr_origin" in src


def test_rewarm_can_be_disabled_for_the_control_arm():
    """Step 2's control arm is the same code with rewarm off."""
    assert "rewarm" in src_check()


def src_check() -> str:
    return open("train.py").read()


def test_schedule_stays_within_bounds():
    """Bounds are [warmup_floor, 1], and the warmup floor is below min_lr_ratio.

    Linear warmup starts at 1/warmup_steps, which is 0.002 here -- well under
    min_lr_ratio 0.1. That is intended: the floor applies to the *decay*, and
    warmup deliberately starts lower. The first version of this test asserted
    min_lr_ratio as the lower bound and failed on the first warmup step.
    """
    for c in (CFG, UPDATE):
        for step in range(0, c["max_steps"] + 10, 7):
            v = base_lr_scale(step, c)
            floor = 1.0 / c["warmup_steps"]
            assert floor - 1e-9 <= v <= 1.0 + 1e-9, (step, v)


def test_min_lr_ratio_is_not_zero_so_the_floor_is_meaningful():
    """If the floor were 0 there would be nothing to re-warm from."""
    assert CFG["min_lr_ratio"] > 0


# --- integration: the two arms must actually differ, verified by running them ---

def _run_pair(tmp_path, rewarm: bool, update_steps: int = 40):
    """Train 60 steps, then resume for `update_steps` more. Returns the LRs."""
    import csv
    import subprocess
    import sys

    import torch

    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name, n, seed in (("train.bin", 20_000, 1), ("val.bin", 8_000, 2)):
        torch.randint(0, 256, (n,), generator=torch.Generator().manual_seed(seed)).to(
            torch.uint16).numpy().tofile(data / name)

    base = yaml.safe_load(open("configs/kalia-v020.yaml"))
    small = {"vocab_size": 256, "n_layer": 2, "n_head": 2, "n_embd": 64, "context_len": 64}

    def write(path, **over):
        c = yaml.safe_load(yaml.safe_dump(base))
        c["model"].update(small)
        c["train"].update(micro_batch_size=4, grad_accum_steps=2, sample_interval=0,
                          log_interval=1, checkpoint_interval_minutes=0.001, **over)
        path.write_text(yaml.safe_dump(c))

    out = tmp_path / ("r" if rewarm else "n")
    write(tmp_path / "a.yaml", max_steps=60, warmup_steps=10, eval_interval=20,
          eval_steps=2, rewarm=rewarm)
    r = subprocess.run(
        [sys.executable, "train.py", "--config", str(tmp_path / "a.yaml"),
         "--data-dir", str(data), "--out-dir", str(out), "--max-steps", "60", "--seed", "5"],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]

    # update_budget, not max_steps: the continual arms ask for "N more steps"
    # while --max-steps keeps its historical absolute meaning.
    write(tmp_path / "b.yaml", max_steps=10, warmup_steps=8, eval_interval=10,
          eval_steps=2, rewarm=rewarm, update_budget=update_steps)
    r2 = subprocess.run(
        [sys.executable, "train.py", "--config", str(tmp_path / "b.yaml"),
         "--data-dir", str(data), "--out-dir", str(out), "--resume", "--seed", "5"],
        capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0, r2.stderr[-2000:]

    rows = list(csv.DictReader((out / "train_log.csv").open()))
    upd = [float(r["lr"]) for r in rows if int(r["step"]) > 60]
    return upd, r2.stdout


def test_resumed_run_actually_trains_the_update_budget(tmp_path):
    """A resumed update trains zero steps unless the budget is origin-relative.

    Found by running a real resume: `step` starts at 60 and `max_steps` was 40,
    so `while step < hard_max_steps` never entered. The run printed "re-warm from
    step 0 of the update budget" and then did nothing.
    """
    lrs, _ = _run_pair(tmp_path, rewarm=True)
    assert len(lrs) > 0, "the resumed run trained no steps"
    assert len(lrs) == 40, f"expected 40 update steps, got {len(lrs)}"


def test_the_two_arms_train_the_same_number_of_steps(tmp_path):
    """Otherwise Step 2 compares an arm that moved against one that did not."""
    a, _ = _run_pair(tmp_path / "a", rewarm=True)
    b, _ = _run_pair(tmp_path / "b", rewarm=False)
    assert len(a) == len(b), f"step counts differ: {len(a)} vs {len(b)}"


def test_the_two_arms_reach_different_learning_rates(tmp_path):
    """The whole point: re-warm climbs, continuation stays at the floor.

    Before the origin was made conditional on `rewarm`, both arms peaked at
    0.00060 -- the control was re-warming too, and Step 2 would have compared two
    identical arms and reported a meaningless null.
    """
    a, out_a = _run_pair(tmp_path / "a", rewarm=True)
    b, out_b = _run_pair(tmp_path / "b", rewarm=False)
    assert max(a) > max(b) * 3, (
        f"re-warm peak {max(a):.6f} should clearly exceed continuation peak "
        f"{max(b):.6f}; if not, the arms are not different"
    )
    assert "re-warm from step 0" in out_a
    assert "continue the source run" in out_b


def test_rewarm_defaults_to_off_so_resume_behaviour_is_unchanged(tmp_path):
    """Every historical run used max_steps as an absolute ceiling.

    With rewarm defaulting to True, a plain --resume would have restarted the
    cosine schedule of every resumed run in the project, changing results that
    were already recorded. It must be opt-in.
    """
    src = open("train.py").read()
    assert 'train_cfg.get("rewarm", False)' in src, (
        "rewarm must default to False so a plain --resume keeps its old behaviour"
    )
    assert 'train_cfg.get("rewarm", True)' not in src


def test_max_steps_keeps_its_historical_absolute_meaning():
    """`--max-steps N` must still mean "stop at absolute step N".

    A smoke test asserts this: resume from step 10 with --max-steps 20 ends at
    step 20. An earlier attempt made every resumed run's budget origin-relative,
    which would have silently changed how v0.2.0 and any already-recorded resume
    behaved. The relative form is a separate config key (`update_budget`) so the
    two meanings cannot be confused.
    """
    assert 'train_cfg.get("update_budget")' in open("train.py").read()
    assert "start_step + int(update_budget)" in open("train.py").read()


def test_update_budget_is_ignored_without_a_resume():
    """A fresh run must not get origin-relative arithmetic against origin 0."""
    src = open("train.py").read()
    assert "if update_budget and start_step:" in src
