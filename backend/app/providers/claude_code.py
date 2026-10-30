"""Claude through the local Claude Code CLI (`claude -p`), billed to the logged-in subscription, so no API key.

For development until a key exists. Each call is one subprocess (~5 s), the subscription's rate limits apply, and
the CLI adds its own wrapper (~1.4k input tokens), so latency and $/JD measured this way do not stand for the API.
Structured output uses the CLI's --json-schema; the schema is a bound kwarg, so it is part of the LLM cache key."""

import asyncio
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

TRANSIENT = re.compile(r"rate.?limit|overloaded|429|529|timed? ?out", re.I)  # retry with backoff
QUOTA = re.compile(r"usage limit|limit reached|resets? at|out of (?:extra )?usage|credit balance", re.I)  # stop
QUOTA_HIT = False  # set once the subscription quota is exhausted; app.eval.run stops instead of saving bad runs


class QuotaExceeded(RuntimeError):
    pass


class ChatClaudeCode(BaseChatModel):
    model: str
    workdir: Path  # empty dir: the CLI must not pick up a project's CLAUDE.md or settings
    timeout: float = 300

    @property
    def _llm_type(self) -> str:
        return "claude-code-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}

    def with_structured_output(self, schema: type[BaseModel], **_: Any):
        return self.bind(json_schema=schema.model_json_schema()) | RunnableLambda(
            lambda m: schema.model_validate_json(m.content))

    def _call(self, messages: list[BaseMessage], json_schema: dict | None) -> tuple[list[str], str]:
        system = "\n\n".join(str(m.content) for m in messages if isinstance(m, SystemMessage))
        rest = [m for m in messages if not isinstance(m, SystemMessage)]
        prompt = str(rest[0].content) if len(rest) == 1 else "\n\n".join(f"[{m.type}]\n{m.content}" for m in rest)
        argv = ["claude", "-p", "--model", self.model, "--output-format", "json", "--tools", "",
                "--no-session-persistence", "--setting-sources", "", "--system-prompt", system or "You are a helpful assistant."]
        if json_schema:
            argv += ["--json-schema", json.dumps(json_schema)]
        self.workdir.mkdir(parents=True, exist_ok=True)
        return argv, prompt

    @staticmethod
    def _env() -> dict[str, str]:
        return {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}  # always the subscription login

    @staticmethod
    def _result(stdout: str, json_schema: dict | None, stderr: str = "") -> ChatResult:
        global QUOTA_HIT
        try:
            d = json.loads(stdout)
        except json.JSONDecodeError:
            d = {"is_error": True, "result": (stderr or stdout).strip()[:300] or "no output"}
        if d.get("is_error") or (json_schema and d.get("structured_output") is None):
            why = str(d.get("result") or d.get("subtype"))
            if QUOTA.search(why):
                QUOTA_HIT = True
                raise QuotaExceeded(f"subscription limit reached: {why}")
            raise RuntimeError(f"claude -p failed: {why}")
        u = d.get("usage", {})
        n_in = u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
        msg = AIMessage(json.dumps(d["structured_output"]) if json_schema else d["result"],
                        usage_metadata={"input_tokens": n_in, "output_tokens": u.get("output_tokens", 0),
                                        "total_tokens": n_in + u.get("output_tokens", 0)},
                        response_metadata={"cost_usd_equivalent": d.get("total_cost_usd")})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _generate(self, messages, stop=None, run_manager=None, json_schema: dict | None = None, **_: Any) -> ChatResult:
        argv, prompt = self._call(messages, json_schema)
        out = subprocess.run(argv, input=prompt, capture_output=True, text=True, cwd=self.workdir, env=self._env(),
                             timeout=self.timeout)
        return self._result(out.stdout, json_schema, out.stderr)

    async def _agenerate(self, messages, stop=None, run_manager=None, json_schema: dict | None = None,
                         **_: Any) -> ChatResult:
        argv, prompt = self._call(messages, json_schema)
        for attempt in range(3):  # transient rate limits and overloads: back off 20 s, then 40 s
            proc = await asyncio.create_subprocess_exec(*argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                        stderr=subprocess.PIPE, cwd=self.workdir, env=self._env())
            try:
                out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), self.timeout)
            except TimeoutError:
                proc.kill()
                raise
            try:
                return self._result(out.decode(), json_schema, err.decode())
            except QuotaExceeded:
                raise
            except RuntimeError as e:
                if attempt == 2 or not TRANSIENT.search(str(e)):
                    raise
                await asyncio.sleep(20 * (attempt + 1))
        raise AssertionError("unreachable")
