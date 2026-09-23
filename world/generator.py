"""Bounded generation and validation repair, independent of transport and engine."""

from dataclasses import dataclass

from pydantic import ValidationError

from llm.prompt_parser import build_messages, repair_message
from llm.providers import LLMBackend
from world.schema import WorldState
from world.validator import WorldValidationError, validate_world


class WorldGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    world: WorldState
    attempts: int


class WorldGenerator:
    def __init__(self, backend: LLMBackend, max_attempts: int = 3):
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        self.backend = backend
        self.max_attempts = max_attempts

    def generate(self, prompt: str) -> GenerationResult:
        messages = build_messages(prompt)
        errors = ""
        for attempt in range(1, self.max_attempts + 1):
            raw = self.backend.complete(messages)
            try:
                world = WorldState.model_validate_json(raw)
                return GenerationResult(validate_world(world), attempt)
            except ValidationError as exc:
                errors = "\n".join(
                    f"{'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}"
                    for e in exc.errors(include_input=False, include_url=False)
                )
            except WorldValidationError as exc:
                errors = str(exc)
            if attempt < self.max_attempts:
                messages.append({"role": "assistant", "content": raw})
                messages.append(repair_message(errors))
        raise WorldGenerationError(
            f"No valid world after {self.max_attempts} attempts: {errors[:6000]}"
        )
