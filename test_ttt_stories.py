"""The X25 probe story set is frozen by registration.

20 Gutenberg passages (unseen by v0.2.0, which never trained on Gutenberg),
each with a fixed 3,000-token prompt and a 1,024-token reference
continuation. If this file changes, the X25 comparison changes with it, so
any edit must go through a ledger amendment — never a quiet rewrite.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STORIES = ROOT / "eval" / "ttt_probe_stories.json"

# sha256 of eval/ttt_probe_stories.json at registration (2026-10-05).
FROZEN_SHA256 = "a94b15f677313b719d26baf59ced1be6ab1b242b988b4488417a6cd7186f7272"


def _load() -> list:
    stories = json.loads(STORIES.read_text())
    assert isinstance(stories, list)
    return stories


def test_story_set_is_frozen():
    digest = hashlib.sha256(STORIES.read_bytes()).hexdigest()
    assert digest == FROZEN_SHA256, (
        "ttt_probe_stories.json changed since registration — "
        "record the change in a ledger amendment, do not edit quietly"
    )


def test_story_set_shape():
    stories = _load()
    assert len(stories) == 20
    sources = set()
    for story in stories:
        assert set(story) == {
            "id",
            "source",
            "prompt",
            "reference_continuation",
            "prompt_tokens",
        }
        assert story["prompt_tokens"] == 3000
        assert len(story["prompt"]) > 5000, "prompt too short to need memory"
        assert len(story["reference_continuation"]) > 1000
        sources.add(story["source"])
    assert len(sources) == 20, "stories must come from distinct sources"
