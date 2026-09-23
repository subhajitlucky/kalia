# KALIA — platform cuts (posting plan)

Canonical piece: the dedicated paper page
`https://subhajitpradhan.vercel.app/kalia` (everything else links back to it).

## 1. LinkedIn (short, hiring-visible)

> I trained a 58M-parameter language model from scratch — random weights, my own
> data pipeline, my own training code — in two days, on free Kaggle GPUs, for $0.
>
> It writes coherent stories, scores 61.4% on PIQA and 45.8% on ARC-Easy
> (competitive with 125M-class models trained on ~160× more tokens), and it beats
> the AdamW baseline's final loss with ~23% fewer tokens after switching to the
> Muon optimizer with QK-Norm.
>
> The part I'm most proud of isn't the model. Every experiment was pre-registered
> with a SHA-256 hash before it ran. A promising optimizer variant won 7 of 7
> checkpoints and still wasn't promoted, because it missed the threshold I set in
> advance. Every mistake is published — 12 incidents, including one where I
> misreported a 44-minute run as five hours.
>
> Full record (model spec, evaluation, replayable training console, all 41
> decisions, all 12 incidents):
> https://subhajitpradhan.vercel.app/kalia
>
> Code: https://github.com/subhajitlucky/kalia
> Weights: https://huggingface.co/kalia-lm/kalia-v012

Posting notes:

- Put the paper-page link in the body, not in a first comment: click-throughs
  matter more than impressions here.
- Attach the console screenshot (dark theme, from
  `/tmp/opencode/kalia-console4.png`) — the replayable-log visual is the
  scroll-stopper.
- Keep the first three lines as the hook; LinkedIn truncates after that.
- Hashtags, sparingly: `#MachineLearning #LLM #Python #FromScratch`.
- Follow-up post when X16 and v0.2.0 land — that is real news, not a repost.

## 2. X / Twitter thread

1/ I trained a 58M-parameter LM from scratch in 2 days on free Kaggle GPUs.
$0 compute. Random init, own tokenizer pipeline, own trainer.
Held-out: 2.44 loss / 0.82 bits-per-byte.
PIQA 61.4% · ARC-Easy 45.8% · HellaSwag 36.8%.
🧵 what worked, what didn't, and the receipts

2/ Constraint: free tier = 30 GPU-h/week on 2×T4, 8.5-h sessions. So the whole
system is built to survive interruption: checkpoints to HF every 30 min, resume
by step counter, no secrets in API runs. ~20 GPU-h total.

3/ Method: micro-ablations before full runs. 30M params, 50M tokens, 30 min per
arm.
AdamW 3.8041 → Muon 3.5937 → +QK-Norm/soft-cap 3.5103 (step-700 val).
Order held at every checkpoint. Then the full run.

4/ The discipline part: Muon+ beat plain Muon on 7/7 checkpoints — by 0.015
nats, under the 0.02 promotion threshold we'd set in advance. It didn't ship.
Thresholds you don't enforce aren't thresholds.

5/ Architecture ablation: looped depth, thin-deep, GQA vs control.
Nothing beat control. Depth-thinning hurt (−0.2 nats); looping/GQA were
quality-neutral.
One finding: the looped model ran 1.35× slower per step, not 2× — reused
weights stay cache-hot.

6/ The training curve lied, then told the truth. Val plateaued for 1,000 steps,
then the cosine decay bit: 2.40. Quota ran out at 73% of schedule. A
deterministic 819k-token eval settled it: 2.4366 — the swings were noise, and
the plateau was real. Stopped under a pre-registered rule.

7/ Coolest metric we invented: the Abhimanyu gap. Reversed-text NLL minus
forward NLL = 6.06 nats. The model can enter fluent text but cannot exit it.
A pre-registered experiment (chunk-preserving reversal training) tests whether
that closes. Named for the warrior who entered the Chakravyuha and couldn't
leave.

8/ Everything is public: 41 numbered decisions, 12 incidents (including my own
misreporting of a run's duration), 2 hash-anchored pre-registrations, all eval
logs.
Code: github.com/subhajitlucky/kalia
Weights: huggingface.co/kalia-lm/kalia-v012
Build log: https://subhajitpradhan.vercel.app/writing/kalia-build-log

## 3. dev.to / Hashnode mirror

- Title: "KALIA: a 58M language model trained from scratch on free GPUs (complete build log)"
- Tags: `machinelearning`, `ai`, `python`, `showdev`
- Canonical URL: point to the portfolio version (dev.to supports `canonical_url`
  in frontmatter).
- Use the full build log text as-is; it is platform-neutral markdown.

## 4. Reddit r/MachineLearning (later, optional)

Title: `[P] KALIA: 58M-parameter LM trained from scratch on free Kaggle GPUs in two days — full decision/incident log, pre-registered experiments, honest negatives`

Body: 4-line summary + the numbers table + links to code/weights/build log.
Post after the portfolio and dev.to are live; reply to comments with the
journal links (the receipts).

## 5. Kaggle discussion (later, optional)

Angle: "What 30 GPU-hours/week buys: a complete 58M training run — with the
notebooks" — link the public ablation + eval kernels and the build log. This is
the audience that cares most about the free-compute engineering.

## Order of operations

1. Portfolio (canonical) → 2. dev.to mirror (canonical tag) → 3. LinkedIn →
4. X thread → 5. Reddit / Kaggle discussion a few days later.
