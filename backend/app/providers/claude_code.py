"""Claude through the local Claude Code CLI (`claude -p`), billed to the logged-in subscription, so no API key.

For development until a key exists. Each call is one subprocess (~5 s), the subscription's rate limits apply, and
the CLI adds its own wrapper (~1.4k input tokens), so latency and $/JD measured this way do not stand for the API.
Structured output uses the CLI's --json-schema; the schema is a bound kwarg, so it is part of the LLM cache key."""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel


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
    def _result(stdout: str, json_schema: dict | None) -> ChatResult:
        d = json.loads(stdout)
        if d.get("is_error") or (json_schema and d.get("structured_output") is None):
            raise RuntimeError(f"claude -p failed: {d.get('result') or d.get('subtype')}")
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
        return self._result(out.stdout, json_schema)

    async def _agenerate(self, messages, stop=None, run_manager=None, json_schema: dict | None = None,
                         **_: Any) -> ChatResult:
        argv, prompt = self._call(messages, json_schema)
        proc = await asyncio.create_subprocess_exec(*argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                    stderr=subprocess.PIPE, cwd=self.workdir, env=self._env())
        try:
            out, _err = await asyncio.wait_for(proc.communicate(prompt.encode()), self.timeout)
        except TimeoutError:
            proc.kill()
            raise
        return self._result(out.decode(), json_schema)
