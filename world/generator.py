"""Bounded generation and validation repair, independent of transport and engine."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from llm.prompt_parser import Message, build_messages, repair_message
from llm.providers import LLMBackend, LLMError
from world.evaluation import GenerationExpectations, ValidationReport, evaluate_response
from world.schema import WorldState


class WorldGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    world: WorldState
    attempts: int


@dataclass(frozen=True)
class GenerationAttempt:
    number: int
    messages: list[Message]
    raw_response: str | None
    validation: ValidationReport


class WorldGenerator:
    def __init__(self, backend: LLMBackend, max_attempts: int = 3):
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        self.backend = backend
        self.max_attempts = max_attempts

    def generate(
        self,
        prompt: str,
        *,
        expectations: GenerationExpectations | None = None,
        on_attempt: Callable[[GenerationAttempt], None] | None = None,
    ) -> GenerationResult:
        messages = build_messages(prompt)
        errors = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = self.backend.complete(messages)
            except LLMError as exc:
                if on_attempt:
                    on_attempt(
                        GenerationAttempt(
                            attempt,
                            deepcopy(messages),
                            exc.raw_response,
                            ValidationReport(
                                "provider_error",
                                errors=[
                                    {
                                        "code": exc.code,
                                        "message": str(exc),
                                    }
                                ],
                            ),
                        )
                    )
                raise
            world, report = evaluate_response(raw, expectations)
            if on_attempt:
                on_attempt(GenerationAttempt(attempt, deepcopy(messages), raw, report))
            if report.passed:
                assert world is not None
                return GenerationResult(world, attempt)
            errors = report.feedback()
            if attempt < self.max_attempts:
                messages.append({"role": "assistant", "content": raw})
                messages.append(repair_message(errors))
        raise WorldGenerationError(
            f"No valid world after {self.max_attempts} attempts: {errors[:6000]}"
        )
