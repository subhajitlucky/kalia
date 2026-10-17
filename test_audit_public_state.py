"""Tests for the pure helpers in tools/audit_public_state.py.

The network probes are calibrated against known-public controls when the tool
runs; what is tested here is the classification logic that decides what those
probes mean, because that logic is what turns a status code into a claim about
exposure.
"""

from __future__ import annotations

from tools.audit_public_state import looks_like_credential, probe_verdict


def test_dataset_verdicts():
    assert probe_verdict("dataset", 403) == "private"
    assert probe_verdict("dataset", 404) == "private"
    assert probe_verdict("dataset", 200) == "PUBLIC"
    assert probe_verdict("dataset", 500) == "unknown"
    assert probe_verdict("dataset", None) == "unknown"


def test_kernel_verdicts():
    assert probe_verdict("kernel", 404) == "private"
    assert probe_verdict("kernel", 200) == "PUBLIC"
    assert probe_verdict("kernel", 401) == "unknown"


def test_bot_challenge_never_counts_as_public():
    challenge = "<title>Checking your browser - reCAPTCHA</title>"
    assert probe_verdict("kernel", 200, challenge) == "unknown"
    assert probe_verdict("dataset", 200, challenge) == "unknown"


def test_credential_shaped_names():
    assert looks_like_credential("hf_token")
    assert looks_like_credential("corpus/.env")
    assert looks_like_credential("my-secret.txt")
    assert not looks_like_credential("tokenizer.json")
    assert not looks_like_credential("generation_config.json")
    assert not looks_like_credential("corpus/tinystories.bin")
