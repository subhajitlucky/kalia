"""Measure the licence mix inside codeparrot-clean — the number I16 needs.

The finding so far, and what is proven versus inferred:

  PROVEN   kalia-prep-v2 (which built the shards v0.1.2 trained on) was created
           in commit 607455e at 12:46:44; the permissive-licence filter landed in
           b50ce5b at 13:26:54. So the shard v0.1.2 trained on was built by the
           pre-filter code, which streamed codeparrot-clean with no licence
           check whatsoever.

  INFERRED codeparrot-clean is deduplicated, not licence-filtered, so it
           contains copyleft and non-permissive files. Plausible from the
           dataset's design and the presence of a `license` field, but NOT yet
           measured.

This script measures it, so the published card can state a number rather than an
inference. The number decides how big the story is: if the non-permissive share is
a fraction of a percent, the honest description is "our filter was slightly out of
date"; if it is a material share, the published model carried a real licence
exposure.

Run as a Kaggle CPU kernel: no GPU quota.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

PERMISSIVE_CODE_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "bsd-2-clause",
    "isc",
    "unlicense",
    "cc0-1.0",
}

# Categories we would never want in a public release, for reporting.
OBVIOUSLY_BAD = {
    "gpl-3.0",
    "gpl-2.0",
    "agpl-3.0",
    "lgpl-3.0",
    "lgpl-2.1",
    "mpl-2.0",
    "cc-by-sa-4.0",
    "cc-by-sa-3.0",
    "cc-by-4.0",
    "cc-by-3.0",
    "epl-2.0",
    "epl-1.0",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20000, help="files to sample")
    parser.add_argument("--out", type=str, default="/kaggle/working/licence_mix.json")
    args = parser.parse_args()

    from datasets import load_dataset

    counts: Counter[str] = Counter()
    chars = Counter()
    seen = 0
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    for row in ds:
        seen += 1
        lic = (row.get("license") or "<none>").strip().lower()
        counts[lic] += 1
        chars[lic] += len(row.get("content") or "")
        if seen >= args.rows:
            break

    total_files = sum(counts.values())
    total_chars = sum(chars.values())
    perm = {k: v for k, v in counts.items() if k in PERMISSIVE_CODE_LICENSES}
    bad = {k: v for k, v in counts.items() if k in OBVIOUSLY_BAD}
    other = {
        k: v
        for k, v in counts.items()
        if k not in PERMISSIVE_CODE_LICENSES and k not in OBVIOUSLY_BAD
    }

    def frac_files(d: dict) -> float:
        return round(sum(d.values()) / total_files, 4) if total_files else 0.0

    def frac_chars(d: dict) -> float:
        return round(sum(chars[k] for k in d) / total_chars, 4) if total_chars else 0.0

    report = {
        "sampled_files": total_files,
        "sampled_chars": total_chars,
        "permissive_licenses": sorted(PERMISSIVE_CODE_LICENSES),
        "permissive": {
            "files": sum(perm.values()),
            "file_share": frac_files(perm),
            "char_share": frac_chars(perm),
            "licenses": dict(sorted(perm.items(), key=lambda kv: -kv[1])[:12]),
        },
        "non_permissive_obvious": {
            "files": sum(bad.values()),
            "file_share": frac_files(bad),
            "char_share": frac_chars(bad),
            "licenses": dict(sorted(bad.items(), key=lambda kv: -kv[1])[:12]),
        },
        "other": {
            "files": sum(other.values()),
            "file_share": frac_files(other),
            "char_share": frac_chars(other),
            "licenses": dict(sorted(other.items(), key=lambda kv: -kv[1])[:20]),
        },
        "top_licenses_overall": dict(counts.most_common(25)),
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(
        "\nNon-permissive share by characters (the number the card should quote): "
        f"{report['non_permissive_obvious']['char_share'] * 100:.2f}%"
    )
    print(f"Plus 'other' (unclassified): {report['other']['char_share'] * 100:.2f}%")


if __name__ == "__main__":
    main()
