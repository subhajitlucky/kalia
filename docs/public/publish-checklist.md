# Publish Checklist — Privacy, Licenses, Claims

Rule: publish the minimum, publish only what we can measure, never publish
anything that could expose private information or misrepresent results. Nothing
goes public until this checklist is fully green.

## 1. Privacy scrub (run before every visibility change)

```bash
# secrets: no tokens or keys anywhere in tracked files
git grep -nE "hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|KGAT_[A-Za-z0-9]+|KAGR" || echo "clean"
# credentials files must stay untracked
git check-ignore -v .secrets/hf_token ~/.kaggle/access_token
# no personal identifiers in docs
git grep -niE "real email|phone|address|password" docs/ README.md || echo "clean"
```

- [ ] No tokens, keys, cookies, or passwords in any tracked file or history
- [ ] No personal contact details; only the chosen public username/org names
- [ ] No chat transcripts are ever published (curated docs only)
- [ ] `.secrets/`, `~/.kaggle/`, `out/`, `data/`, `*.bin`, `*.pt` remain ignored
- [ ] Git history reviewed for anything previously committed by mistake

## 2. Data licenses (state and attribute; never re-publish data)

| Source | License | Redistribution policy |
|---|---|---|
| TinyStories | CDLA-Sharing-1.0 | Do **not** re-publish raw or tokenized data. Attribute. |
| FineWeb-Edu (SmolLM corpus) | ODC-By-1.0 | Do **not** re-publish. Attribute (ODC-By requires attribution). |
| Cosmopedia v2 (SmolLM corpus) | ODC-By-1.0 | Do **not** re-publish. Attribute. |
| Code (CodeParrot-Clean) | Per-file; metadata absent | **Filter to permissive licenses only** (MIT, Apache-2.0, BSD-2/3, ISC, Unlicense, CC0) — implemented in `prepare.py::is_permissive_license` |
| GPT-2 BPE tokenizer | MIT (OpenAI) | Attribute. |

- [ ] Tokenized datasets (`*.bin`) are **never** uploaded publicly
- [ ] Model card contains an attribution section listing every source + license
- [ ] Code subset restricted to permissive licenses only
- [ ] Kaggle notebooks may be published (they contain code, not data)
- [ ] Weights: released under Apache-2.0 with attribution; the card states that
      data licenses apply to the data, not the weights, and lists them plainly

## 3. Claims review (no fake results)

- [ ] Every number in any public text links to a dated journal entry or log
- [ ] Comparisons are same-token, same-data, same-seed where stated
- [ ] No "beats GPT/1B models" claims; only measured, scoped statements
- [ ] Negative results and incidents included (they are the credibility)
- [ ] Limitations section written before publishing, not after criticism

## 4. Name and affiliation

- [ ] Model card states: "KALIA is a personal research project. Not affiliated
      with any government scheme, company, or other project using a similar
      name."
- [ ] No official-sounding claims about safety, reliability, or fitness

## 5. Visibility changes (do at milestone, not before)

- [ ] GitHub repo public **after** v0.1.2 completes
- [ ] HF model repo public **with** the finished model card on the same day
- [ ] Kaggle notebooks public a few days later (ablation notebook first — it is
      the most useful to others)
- [ ] Portfolio article last, linking GitHub + HF + Kaggle

## 6. Rollback plan

- [ ] Know how to flip each repo back to private (HF, GitHub, Kaggle)
- [ ] Keep the ability to delete released weights if a licensing concern arises
