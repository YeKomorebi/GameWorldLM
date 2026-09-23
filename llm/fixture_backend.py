"""Explicit offline regression backend; never silently replaces a live API."""

from examples.scenarios import fixture_worlds, load_scenarios
from llm.prompt_parser import Message
from llm.providers import LLMError


class FixtureBackend:
    def __init__(self):
        worlds = fixture_worlds()
        self.responses = {
            scenario.prompt: worlds[scenario.name].model_dump_json()
            for scenario in load_scenarios()
        }

    def complete(self, messages: list[Message]) -> str:
        prompt = next(message["content"] for message in messages if message["role"] == "user")
        if prompt not in self.responses:
            raise LLMError(
                "Fixture mode accepts only the five exact example prompts; use a live provider"
            )
        return self.responses[prompt]
