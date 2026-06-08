"""
Custom lm-evaluation-harness model wrapper for ollama.

Queries the ollama HTTP API for all inference operations.  Tries the
OpenAI-compatible /v1/completions endpoint with echo first (single-call
loglikelihood); falls back to token-by-token scoring via /api/generate.
"""

import logging
import os
from typing import Optional

import requests as http_requests
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
DEFAULT_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


@register_model("ollama")
class OllamaLM(LM):
    """lm-eval-harness wrapper that delegates to a running ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_gen_toks: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self._max_gen_toks = max_gen_toks
        self._session = http_requests.Session()
        self._echo_ok: Optional[bool] = None
        logger.info("OllamaLM: model=%s  url=%s", model, base_url)

    # ── low-level API calls ──────────────────────────────────────────

    def _api_generate(
        self,
        prompt: str,
        num_predict: int = 128,
        stop: list[str] | None = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        temperature: float = 0,
    ) -> dict:
        body: dict = {
            "model": self.model_name,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": temperature},
        }
        if stop:
            body["options"]["stop"] = stop
        if logprobs:
            body["logprobs"] = True
            if top_logprobs:
                body["top_logprobs"] = top_logprobs
        r = self._session.post(
            f"{self.base_url}/api/generate", json=body, timeout=600
        )
        r.raise_for_status()
        return r.json()

    def _api_completions(
        self,
        prompt: str,
        max_tokens: int = 0,
        echo: bool = True,
        logprobs: int = 5,
    ) -> dict:
        body = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "echo": echo,
            "logprobs": logprobs,
            "temperature": 0,
        }
        r = self._session.post(
            f"{self.base_url}/v1/completions", json=body, timeout=600
        )
        r.raise_for_status()
        return r.json()

    def _check_echo(self) -> bool:
        """Probe whether /v1/completions supports echo + logprobs."""
        if self._echo_ok is not None:
            return self._echo_ok
        try:
            data = self._api_completions("Hi", max_tokens=0, echo=True, logprobs=1)
            lp = data.get("choices", [{}])[0].get("logprobs")
            self._echo_ok = bool(
                lp
                and lp.get("tokens")
                and lp.get("token_logprobs") is not None
            )
        except Exception:
            self._echo_ok = False
        logger.info("Echo-mode available: %s", self._echo_ok)
        return self._echo_ok

    # ── loglikelihood strategies ─────────────────────────────────────

    def _ll_echo(self, context: str, continuation: str) -> tuple[float, bool]:
        """Fast path: single /v1/completions call with echo=true."""
        data = self._api_completions(
            context + continuation, max_tokens=0, echo=True, logprobs=5
        )
        lp = data["choices"][0]["logprobs"]
        tokens = lp["tokens"]
        tok_lps = lp["token_logprobs"]
        top_list = lp.get("top_logprobs") or []

        built, start = "", 0
        for i, t in enumerate(tokens):
            if len(built) >= len(context):
                start = i
                break
            built += t
        else:
            start = len(tokens)

        if start >= len(tokens):
            return -1e10, False

        total = 0.0
        greedy = True
        for i in range(start, len(tokens)):
            if tok_lps[i] is not None:
                total += tok_lps[i]
            if i < len(top_list) and top_list[i]:
                best = max(top_list[i], key=lambda k: top_list[i][k])
                if best != tokens[i]:
                    greedy = False
        return total, greedy

    @staticmethod
    def _parse_gen_logprobs(data: dict):
        """Extract (text, logprob, {alt: logprob}) from /api/generate."""
        text = data.get("response", "")
        raw = data.get("logprobs")
        if not raw or not text:
            return text, None, {}

        content = raw.get("content") if isinstance(raw, dict) else raw
        if isinstance(content, list) and content:
            e = content[0]
            lp = e.get("logprob")
            alts = {
                a["token"]: a["logprob"]
                for a in e.get("top_logprobs", [])
                if "token" in a and "logprob" in a
            }
            return text, lp, alts
        return text, None, {}

    def _ll_generate(self, context: str, continuation: str) -> tuple[float, bool]:
        """Slow path: token-by-token forced decoding via /api/generate."""
        total = 0.0
        greedy = True
        prompt = context
        consumed = 0
        limit = len(continuation) * 4 + 20

        for _ in range(limit):
            if consumed >= len(continuation):
                break
            data = self._api_generate(
                prompt,
                num_predict=1,
                logprobs=True,
                top_logprobs=20,
                temperature=0,
            )
            gen, lp, alts = self._parse_gen_logprobs(data)
            remaining = continuation[consumed:]

            if not gen:
                total -= 100.0
                greedy = False
                prompt += remaining[:1]
                consumed += 1
                continue

            if remaining.startswith(gen):
                total += lp if lp is not None else -5.0
                consumed += len(gen)
                prompt += gen
            else:
                hit = False
                for tok in sorted(alts, key=lambda k: -alts[k]):
                    if tok and remaining.startswith(tok):
                        total += alts[tok]
                        consumed += len(tok)
                        prompt += tok
                        greedy = False
                        hit = True
                        break
                if not hit:
                    total -= 100.0
                    greedy = False
                    step = max(1, len(gen))
                    prompt += remaining[:step]
                    consumed += step

        return total, greedy

    def _compute_ll(self, ctx: str, cont: str) -> tuple[float, bool]:
        if self._check_echo():
            return self._ll_echo(ctx, cont)
        return self._ll_generate(ctx, cont)

    # ── required LM interface ────────────────────────────────────────

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        out: list[tuple[float, bool]] = []
        for req in tqdm(requests, desc="loglikelihood"):
            ctx, cont = req.args
            try:
                res = self._compute_ll(ctx, cont)
            except Exception as exc:
                logger.warning("loglikelihood error: %s", exc)
                res = (-1e10, False)
            self.cache_hook.add_partial("loglikelihood", req.args, res)
            out.append(res)
        return out

    def loglikelihood_rolling(self, requests) -> list[float]:
        out: list[float] = []
        for req in tqdm(requests, desc="loglikelihood_rolling"):
            (text,) = req.args
            try:
                ll, _ = self._compute_ll("", text)
            except Exception as exc:
                logger.warning("loglikelihood_rolling error: %s", exc)
                ll = -1e10
            self.cache_hook.add_partial("loglikelihood_rolling", req.args, ll)
            out.append(ll)
        return out

    def generate_until(self, requests) -> list[str]:
        out: list[str] = []
        for req in tqdm(requests, desc="generate_until"):
            ctx, kwargs = req.args
            until = kwargs.get("until", [])
            max_toks = kwargs.get("max_gen_toks", self._max_gen_toks)

            data = self._api_generate(
                prompt=ctx,
                num_predict=max_toks,
                stop=until or None,
                temperature=kwargs.get("temperature", 0),
            )
            text = data.get("response", "")
            for s in until:
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]

            self.cache_hook.add_partial("generate_until", req.args, text)
            out.append(text)
        return out
