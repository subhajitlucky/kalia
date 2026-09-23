from eval_entities import (
    cooccurrence_edges,
    entity_report,
    extract_entities,
    max_span_ratio,
    retention,
)


def test_extract_entities_basic():
    text = "Lily went to the park. Tom saw Lily. Lily smiled at Tom."
    entities = extract_entities(text)
    assert entities == ["Lily", "Tom"]


def test_extract_entities_sentence_initial_filtered():
    text = "The day was nice."
    assert extract_entities(text) == []


def test_retention_half():
    prompt = "Lily and Tom went outside."
    continuation = "Lily ran to the trees and laughed."
    assert retention(prompt, continuation) == 0.5


def test_retention_full_when_no_entities():
    assert retention("Hello there.", "Nothing here.") == 1.0


def test_max_span_ratio_persistence():
    text = "Lily started. Then she walked. Finally Lily rested."
    ratio = max_span_ratio(text, "Lily")
    assert 0 < ratio < 1


def test_cooccurrence_edges():
    text = "Lily and Tom played. Tom and Sam ran. Lily was happy."
    edges = cooccurrence_edges(text)
    assert ("Lily", "Tom") in edges
    assert ("Sam", "Tom") in edges


def test_entity_report_shape():
    report = entity_report("Lily met Tom.", "Lily and Tom played together.")
    assert report["retention"] == 1.0
    assert "Lily" in report["story_entities"]
    assert isinstance(report["cooccurrence_edges"], list)
