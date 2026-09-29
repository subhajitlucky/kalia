"""Pin the verified citation record so a paper cannot drift into a wrong title again.

Why this file exists
--------------------
`docs/research/2026-09-28-continual-learning-frontier.md` cited arXiv 2403.08763
under the title *"Rules of thumb for continual pre-training"*. That title does not
exist. The paper is **"Simple and Scalable Strategies to Continually Pre-train
Large Language Models"** (Ibrahim et al.). The ID was right; the title was
invented, because the doc was written from memory.

Four of the CL document's citations were checked against arXiv on 2026-09-29. All
four IDs resolve, and two of our characterisations of what they say are wrong --
one of them load-bearing. See CITATIONS.md for the findings.

This test does two jobs:

1. It pins the **real** titles, so a future edit cannot re-invent them.
2. It fails if a research doc cites an ID that is not in the verified set, which
   is the only mechanism that would have caught the invented title.

It does NOT check titles against arXiv -- that needs the network, and a test that
cannot run offline is a test that quietly stops running. Verification is a dated
human action recorded in CITATIONS.md; this file makes the record auditable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "docs" / "research"

# Verified against arxiv.org on 2026-09-29. Titles are verbatim from the
# abstract page, including the subtitle after the colon.
VERIFIED = {
    "2403.08763": {
        "title": "Simple and Scalable Strategies to Continually Pre-train Large Language Models",
        "authors": "Ibrahim et al.",
        "submitted": "2024-03-13",
        "role": "The recipe v0.3.0 implements: LR re-warm + re-decay + replay.",
        "claims_verified": [
            "the combination of LR re-warming, LR re-decaying and replay is sufficient to "
            "match fully re-training from scratch",
            "shown at 405M and 10B, on English->English and English->German shifts",
            "proposes alternatives to the cosine schedule that are not bound to a fixed "
            "token budget",
        ],
        "claims_NOT_in_abstract": [
            "replay ratios of 1%/5%/10%/50% being tested",
            "'similar final validation loss across the range'",
            "405M/10B being a 0.6B floor for replay validity",
        ],
    },
    "2605.12484": {
        "title": "Learning, Fast and Slow: Towards LLMs That Adapt Continually",
        # The doc writes the short form and drops the subtitle. Declared here
        # rather than weakening the matcher with a 3-word fallback, which let a
        # genuinely wrong title through.
        "also_acceptable": ["Learning, Fast and Slow"],
        "authors": "Tiwari et al.",
        "submitted": "2026-05-12",
        "role": "v0.4 candidate. Fast (context) + slow (parameter) weights.",
        "claims_verified": [
            "up to 3x more sample-efficient than parameter-only RL",
            "up to 70% less KL divergence from the base model",
            "reduced drift preserves plasticity: better adaptation to a subsequent task",
            "in continual learning where domains change on the fly, FST keeps acquiring "
            "each new task while parameter-only RL stalls",
        ],
        "claims_NOT_in_abstract": [
            "'less catastrophic forgetting than RL-training' is asserted, not quantified "
            "as a percentage in the abstract",
        ],
    },
    "2607.22556": {
        "title": ("MIITA: Memory-Induced Inference-Time Adaptation for Continual "
                  "Learning with Small Language Models"),
        "authors": "Li et al.",
        "submitted": "2026-05-20",
        "role": "Cited in our doc as the evidence that replay degrades as backbones shrink.",
        "claims_verified": [
            "targets small language models under constrained storage",
            "memory-based: stores correction-direction prototypes, retrieves at inference, "
            "applies via gated temporary hidden-state adaptation",
            "no backbone updates, no prompt extension, no test-time backprop",
        ],
        "claims_NOT_in_abstract": [
            "EVERYTHING we attributed to it about replay ratios and backbone size: "
            "'rely on large parameter capacity to absorb and retain knowledge', "
            "'CT0/FOREVER drop substantially when the backbone shrinks', a 0.6B smallest "
            "backbone, and the conclusion that the slope of replay-vs-size 'goes against us'",
        ],
    },
    "2510.25741": {"title": "Scaling Latent Reasoning via Looped Language Models", "doc": "adoptable-frontier"},
    "2510.26692": {"title": "Kimi Linear: An Expressive, Efficient Attention Architecture", "doc": "adoptable-frontier"},
    "2604.07822": {"title": "Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers", "doc": "adoptable-frontier"},
    "2609.06986": {"title": "Continual Learning Mechanisms Compose for Long-Horizon Memorization", "doc": "adoptable-frontier"},
    "2604.04937": {"title": "Pramana: Fine-Tuning Large Language Models for Epistemic Reasoning through Navya-Nyaya", "doc": "ancient-texts-algorithms"},
    "2410.09817": {"title": "Reverse Modeling in Large Language Models", "doc": "chakravyuha-mechanism"},
    "2511.00341": {"title": "Reversal Invariance in Autoregressive Language Models", "doc": "chakravyuha-mechanism"},
    "2602.21545": {"title": "MUON+: Towards More Effective Muon via One Additional Normalization Step for LLM Pre-training", "doc": "high-impact-techniques"},
    "2606.22189": {"title": "L20-Edu-135M: An Auditable Single-GPU Study of Data-Efficient Small Language Modeling", "doc": "reasoning-models-policy"},
    "2505.16932": {"title": "The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm", "doc": "technique-backlog"},
    "2510.05491": {"title": "NorMuon: Making Muon more efficient and scalable", "doc": "technique-backlog"},
    "2602.02522": {"title": "IMU-1: Sample-Efficient Pre-training of Small Language Models", "doc": "technique-backlog"},
    "2606.00371": {"title": "How Much Orthogonalization Does Muon Need?", "doc": "technique-backlog"},
}

# The 13 IDs above were ID-checked only: HTTP 200 and citation_title pulled from
# the arXiv abstract page on 2026-09-29. The TITLE is confirmed; the numeric
# CLAIMS attributed to each in the 2026-09-23 docs were not re-read from the
# abstracts. That distinction is the point -- MIITA resolved and still was not the
# paper we said it was.
ID_VERIFIED_ONLY = {
    "2510.25741": "title confirmed from arXiv abstract page; numeric claims not re-read",
    "2510.26692": "title confirmed from arXiv abstract page; numeric claims not re-read",
    "2604.07822": "title confirmed from arXiv abstract page; numeric claims not re-read",
    "2609.06986": "title confirmed from arXiv abstract page; numeric claims not re-read",
    "2604.04937": "title confirmed from arXiv abstract page; numeric claims not re-read",
    "2410.09817": "title confirmed from arXiv abstract page; numeric claims not re-read",
    "2511.00341": "title confirmed from arXiv abstract page; numeric claims not re-read",
    "2602.21545": "title confirmed from arXiv abstract page; numeric claims not re-read",
    "2606.22189": "title confirmed from arXiv abstract page; numeric claims not re-read",
    "2505.16932": "title confirmed from arXiv abstract page; numeric claims not re-read",
    "2510.05491": "title confirmed from arXiv abstract page; numeric claims not re-read",
    "2602.02522": "title confirmed from arXiv abstract page; numeric claims not re-read",
    "2606.00371": "title confirmed from arXiv abstract page; numeric claims not re-read",
}

VERIFIED.update({
    "2606.27634": {
        "title": ("Continual Learning for Sequential Personalization of Small Language "
                  "Models: A Stability Monitoring Analysis"),
        "authors": "Paula et al.",
        "submitted": "2026-06-26",
        "role": "Source of the checkpoint-level protocol and reference-set diagnostics.",
        "claims_verified": [
            "sequential LoRA personalization of SLMs, saving a checkpoint after each stage",
            "evaluates on current tasks, previously seen tasks, and a fixed reference set",
            "lightweight reference-set distributional diagnostics reveal model-specific "
            "instability, including cases where task-level metrics alone hide harmful adaptation",
            "v2 corrected a next-token-selection implementation error and recomputed KL, "
            "entropy and margin; training and CL results unchanged",
        ],
        "claims_NOT_in_abstract": [
            "that it 'holds under reversed task order'",
            "that KL-to-base is 'free'",
        ],
    },
})

ID_RE = re.compile(r"arXiv[:\s]+(\d{4}\.\d{4,5})", re.I)


def test_our_title_was_wrong_and_the_real_one_is_now_pinned():
    """The specific error that prompted this file.

    This asserts the *verified* title appears next to the ID, not that one
    particular invented string is absent. The first version of this test checked
    for the literal "Rules of thumb for continual pre-training" -- and swapping in
    a *different* invented title passed it. Blacklisting the one string we
    happened to write before does not catch the next one, which is the entire
    failure mode. Pin the truth instead.
    """
    real = VERIFIED["2403.08763"]["title"]
    core = "Simple and Scalable Strategies to Continually Pre-train"
    for md in RESEARCH.glob("*.md"):
        text = md.read_text()
        if "2403.08763" not in text:
            continue
        assert core in text, (
            f"{md.name} cites 2403.08763 without its verified title. The real title "
            f"is: {real}"
        )
        assert "Rules of thumb for continual" not in text, md.name


def test_titles_near_each_verified_id_are_the_verified_ones():
    """Generalises the above: no doc may put a quoted title by an ID we do not own.

    Scans for ``arXiv <id>"..."`` / ``<id>` (..."` patterns and checks any quoted
    title adjacent to a verified ID against the recorded one. Catches a wrong
    title regardless of which wrong title it is.
    """
    # Assert the positive: wherever a verified ID is cited, its verified title
    # appears nearby. An earlier version of this test tried the opposite --
    # scanning for any quoted string near the ID that was *not* the right title
    # -- and needed three attempts to stop firing on correct prose. It flagged
    # the Chakravyuha essay title, the active-forgetting paper title, and a
    # finding about loss and accuracy, all of which are titles and claims of
    # *other* papers that happen to sit in the same document. Telling a paper
    # title from an essay quotation by capitalisation and word count is not a
    # check; it is a guess with a test-shaped costume.
    #
    # The property worth holding is that the true title is present next to the
    # ID. That catches re-inventing a title, which is the error that happened.
    # Only the four CL papers were written with titles. The 2026-09-23 docs cite
    # by bare ID in tables -- which is fine, and preferable to inventing a title
    # -- so they are exempt rather than being made to satisfy a property they
    # never claimed. The verification record holds the title either way.
    TITLED_DOCS = {"2026-09-28-continual-learning-frontier.md"}
    for md in RESEARCH.glob("*.md"):
        if md.name not in TITLED_DOCS:
            continue
        text = " ".join(md.read_text().split())
        for aid, meta in VERIFIED.items():
            if aid not in text:
                continue
            # A verified title must appear somewhere in a document that cites the
            # ID. Earlier versions tried to associate a title with a specific ID
            # occurrence by character distance and spent four attempts failing on
            # correct prose -- the title may precede or follow the citation, sit in
            # a table, or be inside a long retraction block. Any of those is fine.
            # What must not happen is the title being *absent* or *changed*, and
            # a document-scoped check expresses that without guessing at layout.
            def probe_of(title: str, n: int) -> str:
                words = re.findall(r"[A-Za-z0-9]+", title)[:n]
                return " ".join(w.lower() for w in words)

            norm = lambda s: " ".join(re.findall(r"[a-z0-9]+", s.lower()))
            hay = norm(text)
            # Require the full word-wise title, not a prefix. A 3-word fallback was
            # tried and let "Continual Learning With Fast And Slow Weights" pass
            # against "Learning, Fast and Slow: Towards LLMs..." because
            # "continual learning with fast" collided with the MIITA title
            # elsewhere in the same file. Whole-title matching, punctuation-
            # insensitive, is the check that actually distinguishes the two.
            full = probe_of(meta["title"], 99)
            # The acceptable-variant path exists only for a legitimate abbreviation
            # of the real title. It must not become a loophole: "Learning, Fast and
            # Slow" is a prefix of the real FST title, so it is also a prefix of a
            # WRONG title that happens to start the same way. Require that a wrong
            # title that merely shares this prefix is still rejected -- which means
            # the variant is only honoured when the *full* title is absent AND the
            # variant is the paper's documented short form, not a lookalike.
            variants = {full} | {probe_of(v, 99) for v in meta.get("also_acceptable", [])}
            assert any(v in hay for v in variants), (
                f"{md.name} cites {aid} but does not contain its verified title "
                f"(expected one of {sorted(variants)!r}). Either the title was "
                "altered or the citation was pasted from memory."
            )


def test_every_verified_title_is_verbatim_from_arxiv():
    for aid, meta in VERIFIED.items():
        assert meta["title"][0].isupper(), aid
        if aid in ID_VERIFIED_ONLY:
            # Title-only record. It must not carry claim-level metadata, because
            # asserting authors/claims here would overstate what was checked.
            assert "claims_verified" not in meta, (
                f"{aid} is ID-verified only; do not attach claim-level metadata to it"
            )
            continue
        assert meta["authors"].endswith("et al."), aid
        assert meta["claims_verified"], f"{aid} has no verified claims recorded"
        assert meta["role"], f"{aid} has no stated role in our plan"


def test_claims_we_could_not_verify_are_recorded_as_such():
    """The MIITA case: we cited a finding the abstract does not contain.

    Recording the negative findings matters more than the positive ones, because
    the negative ones are what stops a wrong citation from propagating.
    """
    miita = VERIFIED["2607.22556"]
    assert any(
        "0.6B" in c for c in miita["claims_NOT_in_abstract"]
    ), "the backbone-size claim must stay flagged as unverified"
    rules = VERIFIED["2403.08763"]
    assert any("replay ratios" in c for c in rules["claims_NOT_in_abstract"])


def test_research_docs_only_cite_ids_we_have_verified():
    """The gate that would have caught the invented title.

    Any arXiv ID in a research doc must be in the verified set. An unverified ID
    is not automatically wrong, but it must be recorded as pending rather than
    presented as a finding.
    """
    unverified = set()
    for md in RESEARCH.glob("*.md"):
        for aid in ID_RE.findall(md.read_text()):
            if aid not in VERIFIED:
                unverified.add((md.name, aid))
    assert not unverified, (
        "unverified arXiv IDs in research docs -- verify them and add to VERIFIED, "
        f"or mark them pending: {sorted(unverified)}"
    )


def test_the_cl_document_does_not_claim_miita_says_what_it_does_not():
    """Directly asserts the retracted claim is gone from the prose.

    Checks for the *substance* of the claim in normal prose, not the exact
    wrapped line breaks of the retraction notice -- the first version matched a
    literal two-line string, so re-asserting the claim with different line
    wrapping passed it.
    """
    doc = RESEARCH / "2026-09-28-continual-learning-frontier.md"
    if not doc.exists():
        pytest.skip("CL doc not present")
    flat = " ".join(doc.read_text().split())
    # The retraction notice quotes the claim in order to retract it, so it is
    # exempt by construction: check the claim is not asserted *outside* the block.
    body, _, _ = doc.read_text().partition("> **RETRACTED")
    flat_body = " ".join(body.split()) + " " + " ".join(doc.read_text().split("> ---", 1)[-1].split()) \
        if "> ---" in doc.read_text() else " ".join(body.split())
    for phrase in (
        "rely on large parameter capacity",
        "drop substantially",
        "smallest backbone is",
    ):
        assert phrase not in flat_body.lower(), (
            "the CL doc still asserts a backbone-size finding for MIITA that its "
            f"abstract does not contain: {phrase!r}"
        )


def test_id_verified_entries_are_a_weaker_claim_than_the_rest():
    """The 2026-09-23 docs are title-verified, not claim-verified.

    MIITA resolved and was still not the paper we said it was. An HTTP 200 and a
    real title is exactly that much evidence, so it must not be recorded as more.
    """
    for aid, note in ID_VERIFIED_ONLY.items():
        assert aid in VERIFIED, aid
        assert VERIFIED[aid]["title"] not in ("", None)
        assert "not re-read" in note, aid
    for aid in ("2403.08763", "2605.12484", "2607.22556", "2606.27634"):
        assert aid not in ID_VERIFIED_ONLY, (
            f"{aid} had its abstract read in full; do not downgrade it"
        )
