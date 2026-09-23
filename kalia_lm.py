"""lm-evaluation-harness adapter for KALIA checkpoints (CPU-friendly).

Usage (inside a Kaggle CPU kernel):

    import lm_eval
    from kalia_lm import KaliaLM
    model = KaliaLM(ckpt="/kaggle/input/.../ckpt.pt", device="cpu")
    results = lm_eval.simple_evaluate(model=model, tasks=["piqa"], limit=500)

Implements the three LM primitives the harness needs: loglikelihood,
loglikelihood_rolling, and a minimal greedy generate_until. No transformers
dependency: the model definition and tokenizer are the project's own.
"""

from __future__ import annotations

import torch
from lm_eval.api.model import LM

from eval_reversibility import load_model


class KaliaLM(LM):
    def __init__(self, ckpt: str, device: str = "cpu", batch_size: int = 1):
        super().__init__()
        self._device = torch.device(device)
        self._batch_size = batch_size
        self.model = load_model(ckpt, self._device)

        import tiktoken

        self.enc = tiktoken.get_encoding("gpt2")
        self._max_length = int(self.model.cfg.context_len)

    # ---- harness interface -------------------------------------------------
    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def eot_token_id(self) -> int:
        return 50256

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self.enc.encode_ordinary(string)

    def tok_decode(self, tokens, **kwargs) -> str:
        return self.enc.decode(list(tokens))

    # ---- internals ---------------------------------------------------------
    def _logits(self, ids: list[int]) -> torch.Tensor:
        x = torch.tensor(ids, dtype=torch.long, device=self._device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)[0]
        return logits[0]

    @staticmethod
    def _args(request):
        return request.args if hasattr(request, "args") else tuple(request)

    # ---- primitives --------------------------------------------------------
    def loglikelihood(self, requests):
        results = []
        for request in requests:
            context, continuation = self._args(request)
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)
            if not cont_ids:
                results.append((0.0, True))
                continue
            if not ctx_ids:
                ctx_ids = [self.eot_token_id]  # pseudo-BOS so the first token has a predictor
            if len(cont_ids) >= self.max_length:
                ctx_ids = [self.eot_token_id]
                cont_ids = cont_ids[: self.max_length - 1]
            elif len(ctx_ids) + len(cont_ids) > self.max_length:
                ctx_ids = ctx_ids[-(self.max_length - len(cont_ids)):]
            ids = ctx_ids + cont_ids
            logits = self._logits(ids)
            start = len(ctx_ids) - 1
            logp = torch.log_softmax(logits[start : start + len(cont_ids)], dim=-1)
            targets = torch.tensor(cont_ids, device=logp.device)
            token_logps = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
            is_greedy = bool((logp.argmax(dim=-1) == targets).all())
            results.append((float(token_logps.sum()), is_greedy))
        return results

    def loglikelihood_rolling(self, requests):
        results = []
        stride = max(1, self.max_length // 2)
        for request in requests:
            (text,) = self._args(request)
            ids = self.tok_encode(text)
            if len(ids) < 2:
                results.append((0.0,))
                continue
            total = 0.0
            for start in range(0, len(ids), stride):
                chunk = ids[start : start + self.max_length]
                if len(chunk) < 2:
                    break
                logits = self._logits(chunk)
                logp = torch.log_softmax(logits[:-1], dim=-1)
                targets = torch.tensor(chunk[1:], device=logp.device)
                token_logps = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
                if start == 0:
                    total += float(token_logps.sum())
                else:
                    total += float(token_logps[stride - 1 :].sum())
            results.append((total,))
        return results

    def generate_until(self, requests):
        results = []
        for request in requests:
            context, gen_kwargs = self._args(request)
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_gen = int(gen_kwargs.get("max_gen_toks", 64))
            ids = self.tok_encode(context)[-self.max_length :]
            generated: list[int] = []
            for _ in range(max_gen):
                logits = self._logits((ids + generated)[-self.max_length :])
                generated.append(int(logits[-1].argmax()))
                text = self.tok_decode(generated)
                if any(stop and stop in text for stop in until):
                    break
            results.append(self.tok_decode(generated))
        return results
