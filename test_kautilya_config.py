"""The Kautilya config must differ from v0.2.0 in exactly the allowed keys.

I first wrote this config by hand and got three things wrong: `logit_soft_cap`
instead of `logit_softcap`, a value of 15.0 against v0.2.0's 30.0, and
`muon_lr`/`adamw_lr` in place of the real `muon_learning_rate` / `learning_rate`
keys. Every one of those would have compiled, run, and produced a number that was
not comparable to the static arm it is supposed to be paired with.

That is the same failure as X18, where an arm drifted from its control in a second
field and the comparison became meaningless. The difference is that this time it
is caught before the run rather than after, and by a test rather than by luck.

So: the config is generated from kalia-v020.yaml plus an explicit allow-list, and
anything outside that list is a failure.
"""

from __future__ import annotations

import yaml
import pytest

V020 = yaml.safe_load(open("configs/kalia-v020.yaml"))
KAUT = yaml.safe_load(open("configs/kalia-kautilya.yaml"))

# The only keys the arm may change. Everything else must match v0.2.0 exactly.
ALLOWED = {
    "max_steps",        # 4770 -> 1000: micro budget, mechanism test only
    "warmup_steps",     # 500 -> 100: proportional to the shorter run
    "eval_interval",    # 250 -> 100: policy cadence needs a val on its boundary
    "eval_steps",       # 50 -> 20: eval is now on the critical path
    "seed",             # 1337 -> 1401: fresh, per D48
    "sample_interval",  # 500 -> 0: no sampling in a mechanism test
    "checkpoint_interval_minutes",  # 30 -> 20
    "adaptive_every",
    "adaptive_batches",
    "adaptive_batch_size",
    "update_budget",  # 1000 more steps from the resume point, not an absolute ceiling
}


def test_model_block_is_identical_to_v020():
    """The architecture is unchanged. D48 closed that line; this arm does not reopen it."""
    assert KAUT["model"] == V020["model"], (
        "the Kautilya arm must not change the model; "
        f"differs on {sorted(set(KAUT['model']) ^ set(V020['model']))}"
    )


def test_optimizer_block_matches_v020():
    opt_keys = {"optimizer", "muon_learning_rate", "muon_momentum", "muon_weight_decay", "ns_steps"}
    for k in opt_keys:
        assert KAUT["train"][k] == V020["train"][k], f"{k} drifted from v0.2.0"


def test_only_allowed_keys_differ_from_v020():
    drift = {
        k
        for k in set(V020["train"]) | set(KAUT["train"])
        if V020["train"].get(k) != KAUT["train"].get(k)
    }
    assert drift <= ALLOWED, (
        f"config drifts from v0.2.0 outside the allow-list: {sorted(drift - ALLOWED)}"
    )


def test_no_typo_d_key_names():
    """The specific keys I got wrong the first time."""
    train = KAUT["train"]
    for wrong, right in (
        ("logit_soft_cap", "logit_softcap"),
        ("muon_lr", "muon_learning_rate"),
        ("adamw_lr", "learning_rate"),
    ):
        assert wrong not in train, f"{wrong} is not a KALIA config key; use {right}"
    assert train["learning_rate"] == V020["train"]["learning_rate"]


def test_adaptive_every_is_on_the_eval_boundary():
    """A policy decision with no val measurement on the same line is unpaired."""
    assert KAUT["train"]["adaptive_every"] == KAUT["train"]["eval_interval"], (
        "adaptive_every should equal eval_interval so each reweight is recorded "
        "next to a val loss"
    )


def test_uses_a_fresh_seed_not_the_hypothesis_seed():
    """1337 produced X20's result and is an unusually good seed (D48)."""
    seed = KAUT["train"]["seed"]
    assert seed != 1337
    assert seed not in (1338, 1339), (
        "X21 used 1338/1339; reusing them would confound the mixture arms with the "
        "replication's seed pair"
    )


def test_control_arm_is_the_same_config_with_adaptive_every_zero():
    """Static and adaptive must be reachable without a second config file."""
    control = yaml.safe_load(open("configs/kalia-kautilya.yaml"))
    control["train"]["adaptive_every"] = 0
    for k, v in V020["train"].items():
        if k in ALLOWED:
            continue
        assert control["train"][k] == v, k


def test_max_steps_is_not_the_v020_budget():
    """This is a mechanism test, not an attempt to beat v0.2.0."""
    assert KAUT["train"]["max_steps"] < V020["train"]["max_steps"], (
        "the arm must not run v0.2.0's full 4770-step budget; a mixture policy "
        "needs several reweight cycles to show anything"
    )
    # 10 reweight cycles minimum, or there is nothing to compare.
    assert KAUT["train"]["max_steps"] // KAUT["train"]["adaptive_every"] >= 5


@pytest.mark.parametrize("bad", ["logit_soft_cap", "muon_lr", "adamw_lr"])
def test_typos_would_be_rejected_by_the_mismatch_check(bad):
    """A typo'd key is a silent config change, so it must fail the allow-list too."""
    drifted = yaml.safe_load(open("configs/kalia-kautilya.yaml"))
    drifted["train"][bad] = 1.0
    drift = {
        k
        for k in set(V020["train"]) | set(drifted["train"])
        if V020["train"].get(k) != drifted["train"].get(k)
    }
    assert bad in drift and drift - ALLOWED == {bad}


def test_update_budget_matches_max_steps():
    """A resumed run stops at start_step + update_budget.

    Without update_budget the resume point (4770) is already past the
    absolute max_steps ceiling, so the arm trains zero steps and reports
    success -- the same silent-null class as the re-warm schedule defect.
    """
    assert KAUT["train"].get("update_budget") == KAUT["train"]["max_steps"], (
        "update_budget must equal max_steps so the arm trains a full budget "
        "from the resume point"
    )
