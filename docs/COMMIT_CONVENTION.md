# Commit message convention for this repository

## Never add an attribution or co-author trailer

**No `Co-Authored-By:` line. No `Generated with:` line. No tool attribution of any
kind in the commit message.**

This repository's own history is the specification: the first 116 commits
(2026-09-22 onward) carry no trailer, and that is the convention. A trailer was
added unilaterally partway through 2026-09-28/29 without checking the existing
history, on 14 commits, and had to be stripped with a `filter-branch` rewrite
before any of them were pushed.

**Before writing a commit message, read the last few.** The rule is not a
general one — it is *this repo's* rule, and it can only be read off the history.
`git log -20 --format='%b'` is the check.

This is the same failure mode that produced four broken LAMBADA kernels in one
afternoon: acting on an assumption about an interface instead of reading it. The
assumption here was about your conventions, which is worse, because the repo was
right there to consult.

## What a commit message in this repo does look like

- A subject line naming the outcome, not the activity:
  `X19: implement the branch_norm arm, and correct a pre-registration that flattered it`
  — not `update model.py`.
- A body that explains **why**, and states what was measured. Numbers, not
  adjectives.
- Corrections are stated as corrections. If a commit or pre-registration was
  wrong, the message says so plainly and says how it was caught.
- Negative results get the same prominence as wins. That is D42, and it applies
  to commit messages as much as to the model card.
