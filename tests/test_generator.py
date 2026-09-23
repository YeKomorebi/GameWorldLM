import copy
import json

import pytest

from llm.providers import LLMError
from world.generator import WorldGenerationError, WorldGenerator


class SequenceBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, messages):
        self.calls.append(copy.deepcopy(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_repairs_json_then_spatial_error(forest):
    invalid = forest.model_dump()
    invalid["objects"][4]["position"] = [31, 23]
    backend = SequenceBackend(["not JSON", json.dumps(invalid), forest.model_dump_json()])
    result = WorldGenerator(backend).generate("A village with 3 houses")
    assert result.world == forest
    assert result.attempts == 3
    assert len(backend.calls) == 3
    assert "out_of_bounds" in backend.calls[2][-1]["content"]
    assert backend.calls[2][1]["content"] == "A village with 3 houses"


def test_exhausted_repair_raises_instead_of_rendering_invalid_world():
    backend = SequenceBackend(["{}", "{}"])
    with pytest.raises(WorldGenerationError, match="after 2 attempts"):
        WorldGenerator(backend, max_attempts=2).generate("A village")
    assert len(backend.calls) == 2


def test_provider_failure_is_not_silently_replaced():
    backend = SequenceBackend([LLMError("authentication failed")])
    with pytest.raises(LLMError, match="authentication failed"):
        WorldGenerator(backend).generate("A village")
    assert len(backend.calls) == 1


@pytest.mark.parametrize("prompt", ["", "  ", "x" * 8001])
def test_invalid_prompt_does_not_call_provider(prompt):
    backend = SequenceBackend([])
    with pytest.raises(ValueError):
        WorldGenerator(backend).generate(prompt)
    assert backend.calls == []
