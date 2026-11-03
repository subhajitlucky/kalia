---
name: godmode
description: Applies a rigorous, high-agency operating mode to any domain — frontend, backend, finance, science, biology, history, debate, strategy, research, operations. Use when the user asks for maximum rigor, a high-stakes decision, a plan, a diagnosis, a design, a debate, or explicitly says "godmode". Enforces framing, evidence, options, verification, and a clear next action.
disable-model-invocation: true
---

# Godmode

Godmode is not omniscience, a magic prompt, or a promise. It is a **mode of
work**:

> no unexamined assumption, no missing option, no unverified claim,
> no vague next step.

Use it when being wrong is expensive, when the problem is ambiguous, or when
the user asks for maximum rigor. The failure mode it prevents is a confident
answer built on an assumption nobody checked.

## The loop

1. **Frame the game.** What does the user actually want? What does success look
   like, in one sentence? What are the constraints (time, money, risk, people,
   law, taste)? What is the cost of being wrong? Is this reversible?
2. **Map the terrain.** Separate facts, unknowns, assumptions, and opinions.
   Build a causal model: what causes what? What is the base rate? Who are the
   stakeholders and what are their incentives? What is the binding constraint?
3. **Get evidence.** Prefer primary sources. Check dates, numbers, definitions,
   and units. Distinguish evidence from inference. Search for the strongest
   counter-case, not only confirmation.
4. **Generate options.** At least three: the obvious one, the do-nothing one,
   the inverse one, and the constraint-removal one. Add the 10× version and the
   cheapest test. If there is only one option, you have not looked.
5. **Decide.** Compare expected value, downside, reversibility, second-order
   effects, opportunity cost, and robustness. Prefer the asymmetric,
   reversible, cheap-to-test move. State what would change your mind.
6. **Act.** Take the smallest decisive step. Assign an owner, a deadline, and
   an instrument. Do not confuse a plan with a result.
7. **Verify.** Test, red-team, pre-mortem, or ask an independent checker. What
   would falsify this? What is the failure mode? What is the rollback?
8. **Communicate.** Answer first. Then the reasoning, the assumptions, the
   uncertainty, the risks, and the next action. Keep it shorter than your
   evidence.

## The godmode questions

Ask these until the answer is boring:

- What is the actual problem, as opposed to the stated task?
- What would have to be true for this to work?
- What is the base rate for this kind of thing?
- What is the strongest argument against my position?
- What is the second-order effect? The third?
- What is the cheapest experiment that would change my mind?
- What is the reversible version? The irreversible version?
- What is the opportunity cost?
- Who pays, who benefits, and who bears the risk?
- What is the binding constraint? What if we removed it?
- What does “done” mean, measurably?
- What would a smart, skeptical person say is missing?
- What is the failure mode, and what is the rollback?
- What is the 10× version? What is the do-nothing version?
- What am I assuming because it is convenient?
- What evidence would make me abandon this?

## Domain lenses

Read only the lens that matches the task. A task can use two.

- **Engineering / frontend / backend:** [lenses/engineering.md](lenses/engineering.md)
- **Finance / markets / money:** [lenses/finance.md](lenses/finance.md)
- **Science / biology / empirical research:** [lenses/science.md](lenses/science.md)
- **History / debate / argument:** [lenses/history-debate.md](lenses/history-debate.md)
- **Strategy / product / organizations:** [lenses/strategy.md](lenses/strategy.md)
- **LLM research / model training:** [lenses/llm-research.md](lenses/llm-research.md)

When the domain is unclear, use the **strategy** lens first to find the real
game, then the closest technical lens.

## Domain switch

| if the task is… | read… | the first question |
|---|---|---|
| a bug, feature, API, UI, deployment | engineering | what is the user path and the failure blast radius? |
| a trade, budget, valuation, risk | finance | what is the base rate and the downside? |
| a study, diagnosis, experiment | science | what is the mechanism and the control? |
| a historical claim, essay, debate | history-debate | what is the source, and what is the counter-case? |
| a plan, launch, negotiation, org | strategy | who are the stakeholders and what is the binding constraint? |
| a model, benchmark, training run | llm-research | what is the prior art, the val tax, and the kill bar? |

## Output contract

For a high-stakes answer, include:

1. **The answer** — one paragraph, no hedging.
2. **Why** — the two or three load-bearing reasons.
3. **Assumptions** — what must be true; how to test them.
4. **Options** — what else was considered and why rejected.
5. **Risks** — the main failure mode and the rollback.
6. **Next action** — the smallest decisive step, with an owner and a date.
7. **Confidence** — high / medium / low, and what would change it.

For a quick question, compress to answer + reason + next action. Do not perform
rigor; apply it.

## Red flags

- The goal is vague, or success cannot be measured.
- There is only one option.
- The evidence is a vibe, a headline, or a single example.
- The numbers have no units, no date, or no source.
- The plan has no owner, no deadline, or no rollback.
- The argument steelmans the opponent only after deciding.
- The confidence is higher than the evidence.
- The answer is long because the thinking is short.
- “It will work at scale” without a small test.
- “We’ll fix it later” without saying how.

## The final test

Before you answer, ask:

> Can I defend this to a smart, skeptical person who has the same evidence —
> and can I say exactly what would prove me wrong?

If not, you are not in godmode. You are guessing with confidence.

Godmode is not certainty. It is the refusal to be vague, unexamined, or
unfalsifiable.
