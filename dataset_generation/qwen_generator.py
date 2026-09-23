"""Adapt the existing audited pipeline; throttle every call, including repair calls."""

import math
import time
from pathlib import Path

from gameworldlm.pipeline import GenerationPipeline, PipelineResult
from llm.providers import OpenAICompatibleBackend, ProviderConfig

from .prompt_generator import PromptSpec


class PacedBackend:
    def __init__(self, backend, requests_per_minute: float):
        if not math.isfinite(requests_per_minute) or requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be finite and positive")
        self.backend = backend
        self.interval = 60 / requests_per_minute
        self.next_call = 0.0

    @property
    def last_metadata(self) -> dict:
        return self.backend.last_metadata

    def complete(self, messages):
        delay = self.next_call - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.next_call = time.monotonic() + self.interval
        return self.backend.complete(messages)


class QwenGenerator:
    def __init__(
        self,
        config: ProviderConfig,
        *,
        max_attempts: int = 3,
        tile_size: int = 16,
        requests_per_minute: float = 20,
    ):
        if config.provider != "qwen":
            raise ValueError("QwenGenerator requires a Qwen provider configuration")
        if not 1 <= max_attempts <= 5 or not 16 <= tile_size <= 48:
            raise ValueError("attempts must be 1..5 and tile_size must be 16..48")
        self.config = config
        self.max_attempts = max_attempts
        self.tile_size = tile_size
        self.backend = OpenAICompatibleBackend(config)
        try:
            self.paced = PacedBackend(self.backend, requests_per_minute)
        except ValueError:
            self.backend.close()
            raise
        self.identity = {
            "provider": "qwen",
            "model": config.model,
            "source": "llm",
            "base_url": config.base_url,
            "max_attempts": max_attempts,
            "tile_size": tile_size,
            "max_tokens": config.max_tokens,
            "timeout_seconds": config.timeout,
            "enable_thinking": config.enable_thinking,
        }

    def generate(self, case: PromptSpec, directory: Path) -> PipelineResult:
        pipeline = GenerationPipeline(
            self.paced,
            provider="qwen",
            model=self.config.model,
            output_dir=directory,
            max_attempts=self.max_attempts,
            tile_size=self.tile_size,
            parameters={
                "max_tokens": self.config.max_tokens,
                "timeout_seconds": self.config.timeout,
                "enable_thinking": self.config.enable_thinking,
            },
        )
        return pipeline.run(case.prompt, case_id=case.id, expectations=case.expectations)

    def close(self) -> None:
        self.backend.close()
