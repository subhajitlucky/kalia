"""The v0.3.0 arms must differ from v0.2.0 and from each other in exactly the
keys their registered claim depends on.

Four arms, four claims, and the claims live entirely in the config. Step 1
measures seed variance (so the only difference must be `seed`). Step 2 tests the
re-warm (so the only difference must be `rewarm`). Step 3 sweeps replay (so the
only difference must be `replay_prob`). If any of those arms drifts in a second
field, the comparison measures something else, which is the X18 failure.

The replay arms must also be seed-matched, since the sweep is a within-seed
comparison and a seed difference would swamp any ratio effect.
"""

from __future__ import annotations

import yaml
import pytest

V020 = yaml.safe_load(open("configs/kalia-v020.yaml"))

ARMS = {
    "v030-ctl-s1401": {"seed": 1401, "rewarm": True},
    "v030-ctl-s1402": {"seed": 1402, "rewarm": True},
    "v030-ctl-s1403": {"seed": 1403, "rewarm": True},
    "v030-rewarm-s1401": {"seed": 1401, "rewarm": True},
    "v030-norewarm-s1401": {"seed": 1401, "rewarm": False},
    "v030-replay-00-s1401": {"seed": 1401, "rewarm": True, "replay_prob": 0.0},
    "v030-replay-10-s1401": {"seed": 1401, "rewarm": True, "replay_prob": 0.10},
    "v030-replay-40-s1401": {"seed": 1401, "rewarm": True, "replay_prob": 0.40},
}

# Budget changes are shared by every arm, so they are not "drift".
SHARED = {"max_steps", "warmup_steps", "eval_interval", "eval_steps",
          "sample_interval", "checkpoint_interval_minutes", "seed", "rewarm",
          "replay_prob", "update_budget"}


def _train(name):
    return yaml.safe_load(open(f"configs/{name}.yaml"))["train"]


@pytest.mark.parametrize("name", sorted(ARMS))
def test_model_block_is_untouched(name):
    assert yaml.safe_load(open(f"configs/{name}.yaml"))["model"] == V020["model"]


@pytest.mark.parametrize("name", sorted(ARMS))
def test_optimizer_block_is_untouched(name):
    t = _train(name)
    for k in ("optimizer", "muon_learning_rate", "muon_momentum", "muon_weight_decay",
              "ns_steps", "learning_rate", "grad_clip", "batch_size"):
        if k in V020["train"]:
            assert t[k] == V020["train"][k], f"{name}: {k} drifted"


@pytest.mark.parametrize("name", sorted(ARMS))
def test_no_unexpected_drift_from_v020(name):
    drift = {k for k in set(V020["train"]) | set(_train(name))
             if V020["train"].get(k) != _train(name).get(k)}
    assert drift <= SHARED, f"{name} drifts outside the allow-list: {sorted(drift - SHARED)}"


@pytest.mark.parametrize("name", sorted(ARMS))
def test_expected_values_are_set(name):
    for k, v in ARMS[name].items():
        assert _train(name).get(k) == v, f"{name}: {k} should be {v}, got {_train(name).get(k)}"


def test_step1_pair_differs_only_in_seed():
    """Step 1 is a seed-variance measurement. Nothing else may differ."""
    trains = [_train(n) for n in ("v030-ctl-s1401", "v030-ctl-s1402", "v030-ctl-s1403")]
    base = trains[0]
    for t in trains[1:]:
        differing = {k for k in set(base) | set(t) if base.get(k) != t.get(k)}
        assert differing == {"seed"}, differing
    assert sorted(t["seed"] for t in trains) == [1401, 1402, 1403]


def test_step2_pair_differs_only_in_rewarm():
    a, b = _train("v030-rewarm-s1401"), _train("v030-norewarm-s1401")
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differing == {"rewarm"}, differing


def test_step3_arms_differ_only_in_replay_prob_and_are_seed_matched():
    names = ["v030-replay-00-s1401", "v030-replay-10-s1401", "v030-replay-40-s1401"]
    trains = [_train(n) for n in names]
    base = trains[0]
    for n, t in zip(names[1:], trains[1:]):
        differing = {k for k in set(base) | set(t) if base.get(k) != t.get(k)}
        assert differing == {"replay_prob"}, f"{n}: {differing}"
    assert len({t["seed"] for t in trains}) == 1, "sweep must be within one seed"
    assert [t["replay_prob"] for t in trains] == [0.0, 0.10, 0.40]


def test_no_arm_reuses_a_spent_seed():
    """1337 produced X20; 1338/1339 were X21's replication."""
    for name in ARMS:
        assert _train(name)["seed"] not in (1337, 1338, 1339), name


def test_every_arm_declares_an_update_budget():
    """Resumed arms need "N more steps"; without it they train zero steps.

    v0.2.0 ends at step 4770. An arm with max_steps 500 and no update_budget
    resumes at 4770, finds `step < hard_max_steps` false on entry, and trains
    nothing while still printing that it is re-warming.
    """
    for name in ARMS:
        assert _train(name).get("update_budget") == 500, name


def test_budget_is_a_micro_budget_not_v020s():
    for name in ARMS:
        assert _train(name)["max_steps"] < V020["train"]["max_steps"], name
        assert _train(name)["max_steps"] == 500, name
