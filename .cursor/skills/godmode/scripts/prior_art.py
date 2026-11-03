#!/usr/bin/env python3
"""Probe OpenAlex and arXiv for a claim. No dependencies.

Usage:
    python scripts/prior_art.py "ordered interventions language model"
    python scripts/prior_art.py "compositional curriculum" "held-out composition"

The skill requires two searches with different phrasings. Read the abstracts of
the closest three; a title match is not a verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

OPENALEX = "https://api.openalex.org/works"
ARXIV = "https://export.arxiv.org/api/query"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


def openalex(query: str, limit: int) -> list[str]:
    url = OPENALEX + "?" + urllib.parse.urlencode({
        "search": query,
        "per-page": limit,
        "select": "title,publication_year,doi",
    })
    data = json.loads(urllib.request.urlopen(url, timeout=30).read())
    out = []
    for work in data.get("results", []):
        year = work.get("publication_year") or "?"
        title = " ".join((work.get("title") or "").split())
        doi = work.get("doi") or ""
        out.append(f"  {year} | {title}" + (f" | {doi}" if doi else ""))
    return out


def arxiv(query: str, limit: int) -> list[str]:
    # Exact phrase first, then a keyword fallback: a long exact phrase often
    # returns nothing even when the literature is close.
    words = [w for w in query.split() if len(w) > 2]
    searches = [f'all:"{query}"']
    if words:
        searches.append(" AND ".join(f"all:{w}" for w in words))
    for search in searches:
        url = ARXIV + "?" + urllib.parse.urlencode({
            "search_query": search,
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        })
        data = urllib.request.urlopen(url, timeout=30).read()
        root = ET.fromstring(data)
        out = []
        for entry in root.findall("a:entry", ARXIV_NS):
            title = " ".join(entry.find("a:title", ARXIV_NS).text.split())
            year = entry.find("a:published", ARXIV_NS).text[:10]
            out.append(f"  {year} | {title}")
        if out:
            return out
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", nargs="+")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    for query in args.queries:
        print(f"\n### {query}")
        for name, fn in (("OpenAlex", openalex), ("arXiv", arxiv)):
            try:
                rows = fn(query, args.limit)
            except Exception as exc:  # network / rate limit / parse
                print(f"  {name}: ERROR {type(exc).__name__}: {exc}")
                continue
            print(f"  {name}:")
            if not rows:
                print("    (no results)")
            for row in rows:
                print(row)

    print(
        "\nRead the closest three abstracts. If a reviewer would say "
        "'this is X applied to Y', class it B+, not A."
    )


if __name__ == "__main__":
    sys.exit(main())
