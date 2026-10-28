"""LLM backends: Ollama (default, local), OpenAI-compatible (vLLM), and a
deterministic heuristic backend for offline tests / CI.

All real backends use temperature 0 + fixed seed + JSON output (NFR-01).
"""
from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from cv_agent.llm.gateway import LLMResponse

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def _post_json(url: str, payload: Dict[str, Any], timeout: float, headers: Optional[Dict[str, str]] = None) -> Dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local/configured endpoint
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMError(f"LLM endpoint {url} unreachable: {exc}") from exc


def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    start = text.find("{")
    if start < 0:
        raise LLMError(f"LLM did not return JSON: {text[:200]!r}")
    body = text[start:]
    try:
        return json.loads(body[: body.rfind("}") + 1])
    except json.JSONDecodeError:
        pass
    try:  # output cut by num_predict: close the open string / brackets and keep what was generated
        return json.loads(_close_truncated(body))
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM did not return JSON: {text[:200]!r}") from exc


def _close_truncated(body: str) -> str:
    """Keep only the complete values of a reply cut by num_predict: cut back to the last ``,`` or closing
    bracket outside a string, then close the open brackets. A half-written value (``"line": 1`` of 12, a cut
    string) is dropped rather than trusted."""
    stack: List[str] = []
    in_str = escaped = False
    cut, cut_stack = 0, []
    for i, ch in enumerate(body):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
            cut, cut_stack = i + 1, list(stack)
        elif ch == ",":
            cut, cut_stack = i, list(stack)
    if not cut:
        raise json.JSONDecodeError("no complete value before truncation", body, 0)
    return body[:cut] + "".join(reversed(cut_stack))


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434", seed: int = 42,
                 timeout: float = 300.0, num_ctx: int = 4096, max_output_tokens: Optional[Dict[str, int]] = None,
                 keep_alive: str = "30m") -> None:
        self.model, self.host, self.seed, self.timeout, self.num_ctx = model, host.rstrip("/"), seed, timeout, num_ctx
        self.max_output_tokens = max_output_tokens or {}
        self.keep_alive = keep_alive  # keep the model loaded between calls (a cold load costs seconds)

    def complete_json(self, task: str, system: str, user: str, context: Dict[str, Any]) -> LLMResponse:
        options = {"temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx}
        if task in self.max_output_tokens:  # cap generation: on CPU, output tokens dominate latency
            options["num_predict"] = self.max_output_tokens[task]
        payload = {"model": self.model, "stream": False, "format": "json", "keep_alive": self.keep_alive,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "options": options}
        last_err: Optional[Exception] = None
        for _ in range(2):  # one retry on malformed JSON
            out = _post_json(f"{self.host}/api/chat", payload, self.timeout)
            content = out.get("message", {}).get("content", "")
            try:
                return LLMResponse(_parse_json(content), int(out.get("prompt_eval_count", 0)),
                                   int(out.get("eval_count", 0)), content)
            except (LLMError, json.JSONDecodeError) as exc:
                last_err = exc
                log.warning("Ollama returned invalid JSON for task %s, retrying", task)
        raise LLMError(f"Ollama invalid JSON after retry: {last_err}")

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as resp:  # noqa: S310
                models = [m["name"] for m in json.loads(resp.read()).get("models", [])]
            return self.model in models
        except (urllib.error.URLError, TimeoutError, ValueError):
            return False


class OpenAICompatBackend:
    """vLLM / any OpenAI-compatible server. API key read from env by the caller (never hardcoded)."""
    name = "openai_compat"

    def __init__(self, model: str, base_url: str, api_key: str = "", seed: int = 42, timeout: float = 300.0) -> None:
        self.model, self.base_url, self.api_key, self.seed, self.timeout = model, base_url.rstrip("/"), api_key, seed, timeout

    def complete_json(self, task: str, system: str, user: str, context: Dict[str, Any]) -> LLMResponse:
        payload = {"model": self.model, "temperature": 0, "seed": self.seed,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        out = _post_json(f"{self.base_url}/v1/chat/completions", payload, self.timeout, headers)
        content = out["choices"][0]["message"]["content"]
        usage = out.get("usage", {})
        return LLMResponse(_parse_json(content), int(usage.get("prompt_tokens", 0)),
                           int(usage.get("completion_tokens", 0)), content)


class HeuristicBackend:
    """Deterministic stand-in for an LLM, driven by structured ``context``.

    Used for unit tests and laptop demos without Ollama. It *simulates* a
    fraction of hallucinated claims (``hallucination_rate``) on the first attempt
    so verify_evidence has something to catch. Never report experiment results
    produced with this backend as LLM results.
    """
    name = "heuristic"

    def __init__(self, hallucination_rate: float = 0.25, position_bias: float = 0.0) -> None:
        self.model = f"heuristic-h{hallucination_rate}"
        self.hallucination_rate = hallucination_rate
        self.position_bias = position_bias

    @staticmethod
    def _u(key: str) -> float:
        return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    def complete_json(self, task: str, system: str, user: str, context: Dict[str, Any]) -> LLMResponse:
        handler = getattr(self, f"_{task}", None)
        if handler is None:
            raise LLMError(f"HeuristicBackend: unknown task {task}")
        data = handler(context)
        tokens_in = (len(system) + len(user)) // 4
        return LLMResponse(data, tokens_in, len(json.dumps(data)) // 4, json.dumps(data))

    def _pairwise(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        a, b = ctx["a"], ctx["b"]
        sa, sb = a["score"] + self.position_bias, b["score"]
        winner, best = ("A", a) if sa >= sb else ("B", b)
        return {"winner": winner, "span": (best.get("spans") or [""])[0]}

    def _planner(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Offline stand-in: follow the rule policy's choice (tests exercise the LLM planner path)."""
        choice = ctx.get("rule_choice") or {"action": "finish", "args": {}}
        return {"action": choice["action"], "args": choice.get("args", {}), "reason": "heuristic: rule choice"}

    def _listwise(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        scores: Dict[str, float] = ctx["scores"]
        return {"ranking": sorted(scores, key=lambda k: -scores[k])}

    def _claims(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        claims = [dict(f) for f in ctx.get("facts", [])]
        if ctx.get("attempt", 0) == 0:
            for missing in ctx.get("missing", []):
                if self._u(f"{ctx.get('cv_id')}|{missing}") < self.hallucination_rate:
                    claims.append({"axis": missing.get("axis", ""), "claim": f"Strong hands-on experience with {missing['value']}",
                                   "span": f"expert in {missing['value']}"})
        return {"claims": claims}


def build_backend(cfg: Dict[str, Any]):
    kind = cfg.get("backend", "heuristic")
    if kind == "ollama":
        return OllamaBackend(model=cfg.get("model", "qwen2.5:7b"), host=cfg.get("host", "http://localhost:11434"),
                             seed=cfg.get("seed", 42), timeout=cfg.get("timeout_s", 300),
                             num_ctx=cfg.get("num_ctx", 4096), max_output_tokens=cfg.get("max_output_tokens"))
    if kind == "openai_compat":
        import os
        return OpenAICompatBackend(model=cfg["model"], base_url=cfg["base_url"],
                                   api_key=os.environ.get(cfg.get("api_key_env", "LLM_API_KEY"), ""),
                                   seed=cfg.get("seed", 42))
    if kind == "heuristic":
        return HeuristicBackend(hallucination_rate=cfg.get("hallucination_rate", 0.25))
    raise ValueError(f"Unknown LLM backend {kind!r}")
