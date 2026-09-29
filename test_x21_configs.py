"""Tests for X21's replication configs.

X21 exists because X20's 0.1358-nat result is 3.1x the largest effect any
architecture arm has produced here, on one seed. A replication is worthless if
the "fresh" arms differ from the originals in anything but the seed -- and that is
not hypothetical: earlier in this project an ablation arm drifted from its control
in a second field and the whole comparison became meaningless.

These assert the configs differ by exactly one key, and that the seeds are the
ones the pre-registration names.
"""

from __future__ import annotations

import pytest
import yaml

BASE = yaml.safe_load(open("configs/micro-base.yaml"))
PAIRS = [
    ("micro-base", 1338),
    ("micro-staticgate", 1338),
    ("micro-base", 1339),
    ("micro-staticgate", 1339),
]

# The pre-registration's R-1..R-4 are loss bars and seed-pairing decisions.
# A config that quietly moves anything else invalidates the pairing.
TRAIN_KEYS = ("seed", "max_steps", "learning_rate", "warmup_steps", "optimizer",
              "micro_batch_size", "grad_accum_steps", "min_lr_ratio")


def _load(arm: str, seed: int) -> dict:
    return yaml.safe_load(open(f"configs/{arm}-s{seed}.yaml"))


@pytest.mark.parametrize("arm,seed", PAIRS)
def test_replication_config_differs_from_its_parent_only_in_seed(arm, seed):
    parent = yaml.safe_load(open(f"configs/{arm}.yaml"))
    cfg = _load(arm, seed)
    assert cfg["train"] == {**parent["train"], "seed": seed}, "train block must be parent + seed"
    assert cfg["model"] == parent["model"], "model block must be untouched"


@pytest.mark.parametrize("arm,seed", PAIRS)
def test_replication_configs_are_paired_against_the_same_control(arm, seed):
    """Each treatment must face a control at its own seed, not a shared one."""
    ctl = _load("micro-base", seed)
    assert ctl["train"]["seed"] == seed
    assert _load(arm, seed)["train"]["seed"] == seed


def test_only_the_seed_moves_versus_control():
    """The treatment differs from control by exactly the two arm flags."""
    for seed in (1338, 1339):
        ctl = _load("micro-base", seed)
        arm = _load("micro-staticgate", seed)
        differing = {k for k in set(ctl["model"]) | set(arm["model"])
                     if ctl["model"].get(k) != arm["model"].get(k)}
        assert differing == {"branch_norm", "static_gate"}
        for key in TRAIN_KEYS:
            assert arm["train"][key] == ctl["train"][key], key


def test_no_replication_config_reuses_the_hypothesis_seed():
    """Seed 1337 generated the hypothesis. Including it would be circular."""
    for arm, seed in PAIRS:
        assert seed != 1337, f"{arm}-s{seed} would reuse the hypothesis seed"


def test_training_budget_unchanged_from_the_x18_x19_x20_arms():
    """Everything except the seed must match the arms that produced the original result."""
    for arm, seed in PAIRS:
        t = _load(arm, seed)["train"]
        assert t["max_steps"] == 763, "ablate.py overrides to 500; base must agree"
        for key in TRAIN_KEYS:
            if key == "seed":
                continue  # the one field that is meant to move
            assert t[key] == BASE["train"][key], f"{arm}-s{seed}:{key}"
        assert t["seed"] == seed
