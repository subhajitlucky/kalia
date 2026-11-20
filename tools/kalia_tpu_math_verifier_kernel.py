"""Kaggle TPU pilot: a small math-answer verifier in JAX/Flax.

This is the first capability module for KALIA 150M, not a replacement for the
base run. It trains a ~3M-parameter transformer classifier on synthetic
arithmetic word problems with correct/incorrect candidate answers. The output
is a verifier checkpoint that can later score best-of-N candidates.

Why synthetic: it is self-contained, license-clean, and exercises the full
TPU/JAX path (data -> model -> optimizer -> checkpoint -> HF upload) without
depending on the 150M PyTorch checkpoint. Once the path is proven, the same
model can be trained on real GSM8K/code traces.

Run on Kaggle TPU (machine_shape TpuV5E8), or locally with --smoke.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

PAD, BOS, EOS = 0, 1, 2
BYTE_OFFSET = 3
VOCAB = 256 + BYTE_OFFSET
MAX_LEN = 160


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------

NAMES = ["Alice", "Bob", "Chen", "Dara", "Eli", "Fatima", "Gita", "Hugo",
         "Iris", "Jae", "Kiran", "Lena", "Milo", "Nadia", "Omar", "Priya",
         "Quinn", "Ravi", "Sara", "Tariq"]
OBJECTS = ["apples", "pencils", "marbles", "books", "coins", "stickers",
           "cards", "shells", "buttons", "candies", "flowers", "tokens",
           "oranges", "paperclips", "stamps", "cookies", "beads", "keys",
           "chips", "pens"]


def _problem(rng: random.Random) -> tuple[str, int, int]:
    """Return (problem text, correct answer, a plausible wrong answer)."""
    name = rng.choice(NAMES)
    other = rng.choice([n for n in NAMES if n != name])
    obj = rng.choice(OBJECTS)
    kind = rng.randrange(6)

    if kind == 0:  # add
        a, b = rng.randint(2, 60), rng.randint(2, 60)
        text = f"{name} has {a} {obj}. {name} buys {b} more. How many {obj} does {name} have now?"
        ans = a + b
        wrong = a - b if a > b else a + b + rng.randint(1, 5)
    elif kind == 1:  # subtract
        a = rng.randint(10, 80)
        b = rng.randint(1, a - 1)
        text = f"{name} has {a} {obj}. {name} gives away {b}. How many {obj} does {name} have left?"
        ans = a - b
        wrong = a + b
    elif kind == 2:  # multiply
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        text = f"{name} has {a} boxes with {b} {obj} in each. How many {obj} are there in total?"
        ans = a * b
        wrong = a + b
    elif kind == 3:  # divide
        b = rng.randint(2, 9)
        ans = rng.randint(2, 12)
        a = b * ans
        text = f"{name} has {a} {obj} and divides them equally among {b} friends. How many does each friend get?"
        wrong = a - b
    elif kind == 4:  # two-step
        a, b = rng.randint(5, 40), rng.randint(2, 30)
        c = rng.randint(1, a + b - 1)
        text = (f"{name} has {a} {obj}, buys {b} more, then gives away {c}. "
                f"How many {obj} does {name} have left?")
        ans = a + b - c
        wrong = a + b + c
    else:  # compare
        a = rng.randint(10, 90)
        b = rng.randint(1, a - 1)
        text = f"{name} has {a} {obj} and {other} has {b}. How many more does {name} have?"
        ans = a - b
        wrong = a + b

    if wrong == ans:
        wrong = ans + rng.choice([-3, -2, -1, 1, 2, 3])
    return text, ans, wrong


def make_examples(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    xs = np.zeros((n, MAX_LEN), dtype=np.int32)
    masks = np.zeros((n, MAX_LEN), dtype=np.float32)
    ys = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        problem, correct, wrong = _problem(rng)
        is_correct = rng.random() < 0.5
        candidate = correct if is_correct else wrong
        text = f"Problem: {problem}\nCandidate answer: {candidate}\nIs the candidate correct?"
        ids = [BOS] + [b + BYTE_OFFSET for b in text.encode("utf-8")[: MAX_LEN - 2]] + [EOS]
        xs[i, : len(ids)] = ids
        masks[i, : len(ids)] = 1.0
        ys[i] = 1.0 if is_correct else 0.0
    return xs, masks, ys


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class Block(nn.Module):
    dim: int
    heads: int

    @nn.compact
    def __call__(self, x, mask):
        b, t, c = x.shape
        h = self.heads
        d = c // h

        y = nn.LayerNorm()(x)
        qkv = nn.Dense(3 * c)(y)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(b, t, h, d)
        k = k.reshape(b, t, h, d)
        v = v.reshape(b, t, h, d)

        attn = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(d)
        attn = jnp.where(mask[:, None, None, :] > 0, attn, -1e9)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhts,bshd->bthd", attn, v).reshape(b, t, c)
        x = x + nn.Dense(c)(out)

        y = nn.LayerNorm()(x)
        y = nn.Dense(4 * c)(y)
        y = nn.gelu(y)
        y = nn.Dense(c)(y)
        return x + y


class Verifier(nn.Module):
    dim: int = 256
    layers: int = 4
    heads: int = 4
    max_len: int = MAX_LEN
    vocab: int = VOCAB

    @nn.compact
    def __call__(self, x, mask):
        x = nn.Embed(self.vocab, self.dim)(x)
        pos = nn.Embed(self.max_len, self.dim)(jnp.arange(x.shape[1]))
        x = x + pos[None, :, :]
        for _ in range(self.layers):
            x = Block(self.dim, self.heads)(x, mask)
        x = nn.LayerNorm()(x)
        m = mask[..., None]
        pooled = (x * m).sum(axis=1) / jnp.maximum(m.sum(axis=1), 1.0)
        y = nn.Dense(self.dim)(pooled)
        y = nn.gelu(y)
        y = nn.Dense(1)(y)
        return y.squeeze(-1)


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-n", type=int, default=200_000)
    ap.add_argument("--val-n", type=int, default=20_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default="/kaggle/working/tpu-verifier")
    ap.add_argument("--hf-repo", default="kalia-lm/kalia-tpu-verifier")
    args = ap.parse_args()

    if args.smoke:
        args.train_n, args.val_n, args.epochs = 2_000, 1_000, 1
        args.batch_size = 64

    print("jax", jax.__version__, "devices", jax.devices(), flush=True)
    if not any(getattr(d, "platform", "") == "tpu" for d in jax.devices()):
        raise SystemExit("no TPU device; refusing to train this pilot on CPU")
    t0 = time.time()
    x_tr, m_tr, y_tr = make_examples(args.train_n, args.seed)
    x_va, m_va, y_va = make_examples(args.val_n, args.seed + 1)
    print(f"data: train {x_tr.shape} val {x_va.shape} in {time.time()-t0:.1f}s", flush=True)

    model = Verifier()
    rng = jax.random.PRNGKey(args.seed)
    variables = model.init(rng, x_tr[:1], m_tr[:1])
    params = variables["params"]
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print("params", n_params, flush=True)

    tx = optax.adamw(args.lr, weight_decay=1e-2)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, x, mask, y):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, x, mask)
            loss = optax.sigmoid_binary_cross_entropy(logits, y).mean()
            return loss
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def eval_step(state, x, mask, y):
        logits = state.apply_fn({"params": state.params}, x, mask)
        loss = optax.sigmoid_binary_cross_entropy(logits, y).mean()
        pred = (logits > 0).astype(jnp.float32)
        acc = (pred == y).mean()
        return loss, acc

    rng = np.random.default_rng(args.seed)
    steps_per_epoch = max(1, args.train_n // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(args.train_n)
        for _ in range(steps_per_epoch):
            idx = order[step % args.train_n : step % args.train_n + args.batch_size]
            if len(idx) < args.batch_size:
                idx = order[: args.batch_size]
            xb = jnp.asarray(x_tr[idx])
            mb = jnp.asarray(m_tr[idx])
            yb = jnp.asarray(y_tr[idx])
            state, loss = train_step(state, xb, mb, yb)
            step += 1
            if step % 100 == 0 or step == total_steps:
                print(f"step {step}/{total_steps} loss {float(loss):.4f} "
                      f"elapsed {time.time()-t0:.1f}s", flush=True)

    val_loss, val_acc = eval_step(state, jnp.asarray(x_va), jnp.asarray(m_va), jnp.asarray(y_va))
    train_loss, train_acc = eval_step(state, jnp.asarray(x_tr[:5000]), jnp.asarray(m_tr[:5000]), jnp.asarray(y_tr[:5000]))
    metrics = {
        "experiment": "tpu-math-verifier-pilot",
        "params": int(n_params),
        "train_n": args.train_n,
        "val_n": args.val_n,
        "steps": int(total_steps),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "elapsed_s": round(time.time() - t0, 2),
        "devices": [str(d) for d in jax.devices()],
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    from flax import serialization
    (out / "verifier_params.msgpack").write_bytes(serialization.to_bytes(state.params))
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)

    token = os.environ.get("HF_TOKEN")
    if not token:
        for p in Path("/kaggle/input").glob("**/token.txt"):
            if "kalia-hf-token" in str(p):
                token = p.read_text().strip()
                break
    if token:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(args.hf_repo, repo_type="model", private=True, exist_ok=True)
        for name in ("metrics.json", "verifier_params.msgpack"):
            api.upload_file(
                path_or_fileobj=str(out / name),
                path_in_repo=f"pilot/{name}",
                repo_id=args.hf_repo,
            )
        print("uploaded to", args.hf_repo, flush=True)
    else:
        print("no HF token; artifacts left in", out, flush=True)


if __name__ == "__main__":
    main()
