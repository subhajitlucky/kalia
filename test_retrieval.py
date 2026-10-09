"""Tests for the lexical lookup module."""

import pytest

from retrieval import BM25Index, augment_prompt, build_context, load_corpus, tokenize

DOCS = [
    "the quick brown fox jumps over the lazy dog",
    "a rabbit runs fast across the field",
    "engine repair manual for small diesel motors",
]


def test_exact_term_match_ranks_first():
    index = BM25Index(DOCS)
    assert index.retrieve("fox")[0][0] == 0
    assert index.retrieve("rabbit")[0][0] == 1
    assert index.retrieve("diesel")[0][0] == 2


def test_rare_terms_outweigh_common_ones():
    docs = ["apple banana", "apple cherry", "apple banana cherry durian"]
    index = BM25Index(docs)
    assert index.retrieve("apple durian")[0][0] == 2


def test_tokenize_ignores_case_and_punctuation():
    assert tokenize("The FOX, jumped!") == ["the", "fox", "jumped"]
    index = BM25Index(DOCS)
    assert index.retrieve("FOX!")[0][0] == 0


def test_ties_and_order_are_stable():
    index = BM25Index(["echo echo", "echo echo"])
    assert [i for i, _ in index.retrieve("echo")] == [0, 1]


def test_blank_queries_return_nothing_rather_than_everything():
    index = BM25Index(DOCS)
    assert index.retrieve("") == []
    assert index.retrieve("!!! ...") == []


def test_empty_or_blank_corpora_are_refused():
    with pytest.raises(ValueError):
        BM25Index([])
    with pytest.raises(ValueError):
        BM25Index(["fine", "   "])


def test_context_respects_the_budget_and_puts_the_best_hit_first():
    index = BM25Index(DOCS)
    context = build_context("fox", index, k=3, max_chars=30)
    assert context.startswith("the quick brown fox")
    assert len(context) <= 30


def test_context_is_empty_when_nothing_matches():
    index = BM25Index(DOCS)
    assert build_context("zebra", index) == ""


def test_augment_without_hits_is_the_query_unchanged():
    index = BM25Index(DOCS)
    assert augment_prompt("zebra", index) == "zebra"
    augmented = augment_prompt("fox", index, max_chars=30)
    assert augmented.endswith("\n\nfox")


def test_load_corpus_reads_txt_and_jsonl(tmp_path):
    txt = tmp_path / "docs.txt"
    txt.write_text("alpha one\n\nbeta two\n")
    assert load_corpus(txt) == ["alpha one", "beta two"]

    jsonl = tmp_path / "docs.jsonl"
    jsonl.write_text('{"text": "gamma three"}\n{"text": "delta four"}\n')
    assert load_corpus(jsonl) == ["gamma three", "delta four"]
