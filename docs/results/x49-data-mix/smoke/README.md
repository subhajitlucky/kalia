# X49 smoke test — control, 50 steps

- Resume v0.3.1, control mixture, 50 steps, warmup 5, cosine.
- Val loss after 50 steps: **2.8085** (base v0.3.1 2.7579; the short high-LR
  re-warm is expected to raise val before it helps).
- Throughput: **13.6k tokens/s** on the 58M model.
- Checkpoint saved; the pipeline works.
- Runtime consequence: 2,000 steps would take ~10.7h, over the 9h session
  limit. Amendment 1 sets both arms to 1,500 steps (~8.0h) in one session.
