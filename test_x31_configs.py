"""Tests for X31's QK-Norm / soft-cap isolation configs.

X31 exists because D8's "validated" status rests on one seed and one bundle.
The comparison is only readable if every condition differs from control by
exactly the two model flags and the seed -- earlier in this project an ablation
arm drifted from its control in a second field and the comparison became
meaningless (X21's tests exist for that reason).

These assert the four conditions are a true 2x2, that the seeds are the fresh
pair the pre-registration names, and that no training knob moved.
"""

from __future__ import annotations

import pytest
import yaml

BASE = yaml.safe_load(open("configs/micro-muon.yaml"))
SEEDS = (1338, 1339)
CONDITIONS = {
    "micro-muon": (False, 0.0),
    "micro-qk": (True, 0.0),
    "micro-cap": (False, 30.0),
    "micro-qkcap": (True, 30.0),
}
PAIRS = [(name, seed) for name in CONDITIONS for seed in SEEDS]


def _load(name: str, seed: int) -> dict:
    return yaml.safe_load(open(f"configs/{name}-s{seed}.yaml"))


@pytest.mark.parametrize("name,seed", PAIRS)
def test_model_block_is_the_control_plus_exactly_the_two_flags(name, seed):
    cfg = _load(name, seed)
    qk, cap = CONDITIONS[name]
    expected_model = {
        **BASE["model"],
        "qk_norm": qk,
        "logit_softcap": cap,
    }
    assert cfg["model"] == expected_model
    differing = {
        key
        for key in set(cfg["model"]) | set(BASE["model"])
        if cfg["model"].get(key) != BASE["model"].get(key)
    }
    assert differing == {"qk_norm", "logit_softcap"}, differing


@pytest.mark.parametrize("name,seed", PAIRS)
def test_train_block_is_the_control_plus_exactly_the_seed(name, seed):
    cfg = _load(name, seed)
    assert cfg["train"] == {**BASE["train"], "seed": seed}


def test_the_four_conditions_are_a_true_2x2():
    observed = {
        name: (cfg["model"]["qk_norm"], cfg["model"]["logit_softcap"])
        for name in CONDITIONS
        for cfg in [_load(name, SEEDS[0])]
    }
    assert observed == CONDITIONS
    assert len({qk for qk, _ in observed.values()}) == 2
    assert len({cap for _, cap in observed.values()}) == 2


def test_no_condition_reuses_the_hypothesis_seed():
    """Seed 1337 generated the D8 result. Including it would be circular."""
    assert 1337 not in SEEDS
    for name, seed in PAIRS:
        assert seed != 1337, f"{name}-s{seed} would reuse the hypothesis seed"


def test_the_bundle_arm_matches_the_existing_bundled_config():
    """`micro-qkcap` should be the old `micro-muon-qk` arm plus a fresh seed."""
    old = yaml.safe_load(open("configs/micro-muon-qk.yaml"))
    for seed in SEEDS:
        cfg = _load("micro-qkcap", seed)
        assert cfg["model"] == old["model"]
        assert cfg["train"] == {**old["train"], "seed": seed}
