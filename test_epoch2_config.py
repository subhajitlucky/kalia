"""The v0.3.0 second-epoch config must differ from v0.2.0 only in the budget,
the re-warm cosine and the evaluation cadence.

This is a model-improvement run, not an architecture or data experiment. If the
model block or an optimizer hyperparameter drifts, the new checkpoint is not a
longer v0.2.0; it is a different model wearing the same name. The drift guard is
the same one X18 forced us to write.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
V020 = yaml.safe_load(open("configs/kalia-v020.yaml"))
EPOCH2 = yaml.safe_load(open("configs/v030-epoch2.yaml"))

ALLOWED = {
    "max_steps",
    "update_budget",
    "warmup_steps",
    "eval_interval",
    "eval_steps",
    "sample_interval",
    "seed",
    "rewarm",
}


def test_model_block_is_identical_to_v020():
    assert EPOCH2["model"] == V020["model"], (
        "the second epoch must not change the model; "
        f"differs on {sorted(set(EPOCH2['model']) ^ set(V020['model']))}"
    )


def test_optimizer_block_matches_v020():
    for k in ("optimizer", "muon_learning_rate", "muon_momentum",
              "muon_weight_decay", "ns_steps", "learning_rate", "min_lr_ratio",
              "micro_batch_size", "grad_accum_steps", "weight_decay",
              "beta1", "beta2", "grad_clip"):
        assert EPOCH2["train"][k] == V020["train"][k], f"{k} drifted from v0.2.0"


def test_only_allowed_keys_differ_from_v020():
    drift = {
        k
        for k in set(V020["train"]) | set(EPOCH2["train"])
        if V020["train"].get(k) != EPOCH2["train"].get(k)
    }
    assert drift <= ALLOWED, (
        f"config drifts from v0.2.0 outside the allow-list: {sorted(drift - ALLOWED)}"
    )


def test_budget_and_rewarm_are_the_registered_ones():
    t = EPOCH2["train"]
    assert t["max_steps"] == 2000, "the re-warm cosine is 2,000 steps"
    assert t["update_budget"] == 2000, "resumed run needs 2,000 more steps"
    assert t["rewarm"] is True, "the second epoch re-warms from the v0.2.0 floor"
    assert t["warmup_steps"] == 200, "10% warmup, proportional to the update"
    assert t["seed"] == 1401, "fresh data order; weights are fixed"
    assert t["eval_steps"] == 100, "canonical 100-batch evaluation"
    assert t["eval_interval"] == 100


def test_schedule_is_cosine_not_the_unmeasured_wsd():
    """WSD is implemented but has never been run at 58M; the model run uses the
    proven re-warm cosine, not an unmeasured schedule change."""
    assert EPOCH2["train"].get("lr_schedule", "cosine") == "cosine"


def test_torchrun_entrypoint_is_train_py_not_the_interpreter():
    """The first v0.3.0 launch passed `sys.executable` as torchrun's entrypoint,
    so the launcher ran `/usr/bin/python3` as a script and died on null bytes.
    The base command must start at train.py; the interpreter belongs only in
    front of `-m torch.distributed.run`."""
    src = (ROOT / "tools" / "kalia_v030_epoch2_kernel.py").read_text()
    assert 'cmd = [\n    "train.py",' in src
    assert 'cmd = [sys.executable, "train.py",' not in src
