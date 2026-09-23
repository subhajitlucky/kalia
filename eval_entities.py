"""Entity-consistency metrics for KALIA generations.

Inspired by computational network analyses of the Mahabharata (Gultepe &
Mathangi, Heritage 2023) and the entity metrics of Papalampidi & Lapata (2022):
measure whether characters introduced in a prompt persist, stay coherent, and
interact consistently in the generated continuation. Heuristic, deterministic,
no external models.
"""

import argparse
import re
from collections import Counter
from pathlib import Path

STOPWORDS = {
    "The", "Once", "When", "Then", "She", "He", "It", "They", "But", "And",
    "One", "Every", "There", "This", "That", "His", "Her", "After", "Before",
    "Soon", "Finally", "So", "At", "In", "On", "With", "As", "If", "Now",
    "Later", "Suddenly", "Meanwhile", "Its", "Their", "Not", "Yes", "No",
    "Outside", "Inside", "Today", "Tomorrow", "Yesterday", "Night", "Morning",
    "Evening", "Here", "While", "Because", "During", "Although", "Two", "Three",
}

_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")


def extract_entities(text: str) -> list[str]:
    """Non-stopword capitalized words (length >= 3), counted anywhere in the text."""
    counts: Counter[str] = Counter()
    for match in _NAME_RE.finditer(text):
        name = match.group(1)
        if name not in STOPWORDS:
            counts[name] += 1
    return sorted(counts)


def retention(prompt_text: str, continuation_text: str) -> float:
    """Fraction of prompt entities that still appear in the continuation."""
    prompt_entities = extract_entities(prompt_text)
    if not prompt_entities:
        return 1.0
    kept = sum(1 for name in prompt_entities if re.search(rf"\b{re.escape(name)}\b", continuation_text))
    return kept / len(prompt_entities)


def max_span_ratio(text: str, entity: str) -> float:
    """(last mention - first mention) / length of text: how long the entity persists."""
    matches = [m.start() for m in re.finditer(rf"\b{re.escape(entity)}\b", text)]
    if len(matches) < 2 or not text:
        return 0.0
    return (matches[-1] - matches[0]) / len(text)


def cooccurrence_edges(text: str) -> set[tuple[str, str]]:
    """Entity pairs appearing in the same sentence (the network view)."""
    entities = set(extract_entities(text))
    edges: set[tuple[str, str]] = set()
    for sentence in re.split(r"[.!?]+", text):
        present = sorted(name for name in entities if re.search(rf"\b{re.escape(name)}\b", sentence))
        for i, left in enumerate(present):
            for right in present[i + 1 :]:
                edges.add((left, right))
    return edges


def entity_report(prompt: str, continuation: str) -> dict:
    prompt_entities = extract_entities(prompt)
    full_text = f"{prompt} {continuation}"
    story_entities = extract_entities(full_text)
    return {
        "prompt_entities": prompt_entities,
        "story_entities": story_entities,
        "retention": round(retention(prompt, continuation), 4),
        "max_span_ratio": {
            name: round(max_span_ratio(full_text, name), 4) for name in prompt_entities
        },
        "cooccurrence_edges": sorted(cooccurrence_edges(full_text)),
    }


def render_report(prompt: str, continuation: str) -> str:
    report = entity_report(prompt, continuation)
    lines = [
        "# Entity Consistency Report",
        "",
        f"- Prompt entities: {', '.join(report['prompt_entities']) or '(none)'}",
        f"- Story entities: {', '.join(report['story_entities']) or '(none)'}",
        f"- Retention: **{report['retention']}**",
        f"- Co-occurrence edges: {len(report['cooccurrence_edges'])}",
        "",
        "| Entity | Max-span ratio |",
        "|---|---|",
    ]
    for name, ratio in report["max_span_ratio"].items():
        lines.append(f"| {name} | {ratio} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("out/eval/entities.md"))
    args = parser.parse_args()
    report = render_report(args.prompt, args.continuation.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
