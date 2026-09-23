"""OpenAI and Qwen share transport, while output modes remain explicit."""

import math
import os
import time
from dataclasses import dataclass, field
from typing import Protocol

from openai import APIError, OpenAI

from llm.prompt_parser import Message
from world.schema import strict_json_schema


class LLMError(RuntimeError):
    def __init__(
        self, message: str, *, code: str = "provider_error", raw_response: str | None = None
    ):
        self.code = code
        self.raw_response = raw_response
        super().__init__(message)


class LLMBackend(Protocol):
    def complete(self, messages: list[Message]) -> str: ...


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    api_key: str = field(repr=False)
    base_url: str
    timeout: float = 60
    max_tokens: int = 8192
    enable_thinking: bool | None = False

    @classmethod
    def from_env(cls, provider: str | None = None, model: str | None = None) -> "ProviderConfig":
        name = provider or os.getenv("GAMEWORLDLM_PROVIDER", "openai")
        if name not in {"openai", "qwen"}:
            raise ValueError("Provider must be openai or qwen")
        if name == "openai":
            key = os.getenv("OPENAI_API_KEY", "")
            base = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            chosen_model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        else:
            key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
            base = os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            chosen_model = model or os.getenv("QWEN_MODEL") or "qwen-plus"
        if not key.strip():
            raise ValueError(f"Missing {name.upper()}_API_KEY. Set it in your environment or .env")
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        budget = int(os.getenv("LLM_MAX_TOKENS", "8192"))
        if not math.isfinite(timeout) or timeout <= 0 or budget < 256:
            raise ValueError("LLM_TIMEOUT_SECONDS must be positive; LLM_MAX_TOKENS must be >=256")
        thinking = os.getenv("QWEN_ENABLE_THINKING", "false").lower() if name == "qwen" else "omit"
        if thinking not in {"false", "omit"}:
            raise ValueError("QWEN_ENABLE_THINKING must be false or omit for JSON generation")
        return cls(
            name, chosen_model, key, base, timeout, budget, None if thinking == "omit" else False
        )


class OpenAICompatibleBackend:
    def __init__(self, config: ProviderConfig, client: OpenAI | None = None):
        self.config = config
        self.last_metadata: dict = {}
        self.client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=1,
        )

    def complete(self, messages: list[Message]) -> str:
        started = time.perf_counter()
        self.last_metadata = {"requested_model": self.config.model}
        if self.config.provider == "openai":
            output_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "world_state",
                    "strict": True,
                    "schema": strict_json_schema(),
                },
            }
            token_options = {"max_completion_tokens": self.config.max_tokens}
        else:
            output_format = {"type": "json_object"}
            token_options = {"max_tokens": self.config.max_tokens}
            if self.config.enable_thinking is not None:
                token_options["extra_body"] = {"enable_thinking": self.config.enable_thinking}
        try:
            result = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                response_format=output_format,
                **token_options,
            )
        except APIError as exc:
            status = getattr(exc, "status_code", None)
            self.last_metadata.update(
                {
                    "http_status": status,
                    "request_id": getattr(exc, "request_id", None),
                    "latency_seconds": round(time.perf_counter() - started, 3),
                }
            )
            raise LLMError(
                f"{self.config.provider} request failed ({type(exc).__name__}, HTTP {status}). "
                "Check API key, endpoint, model access, quota, and network.",
                code=type(exc).__name__,
            ) from exc
        self.last_metadata.update(
            {
                "response_id": result.id,
                "actual_model": result.model,
                "request_id": getattr(result, "_request_id", None),
                "usage": result.usage.model_dump() if result.usage else None,
                "latency_seconds": round(time.perf_counter() - started, 3),
            }
        )
        if not result.choices:
            raise LLMError("Provider returned no completion choices")
        choice = result.choices[0]
        self.last_metadata["finish_reason"] = choice.finish_reason
        if choice.message.refusal:
            raise LLMError(
                "Provider refused the scene request",
                code="refusal",
                raw_response=choice.message.content,
            )
        if choice.finish_reason != "stop":
            raise LLMError(
                f"Incomplete response: {choice.finish_reason}. "
                "For length, increase LLM_MAX_TOKENS.",
                code="incomplete_response",
                raw_response=choice.message.content,
            )
        if not choice.message.content:
            raise LLMError("Provider returned an empty response")
        return choice.message.content

    def close(self) -> None:
        self.client.close()
