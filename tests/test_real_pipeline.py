import json
from copy import deepcopy

import pytest
from PIL import Image

from examples.real_generation import load_cases
from gameworldlm.pipeline import GenerationPipeline
from llm.providers import LLMError
from world.evaluation import GenerationExpectations, evaluate_response
from world.io import load_world


class RecordedBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.last_metadata = {}

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        self.last_metadata = {
            "actual_model": "unit-test-model",
            "request_id": f"request-{len(self.calls)}",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def make_pipeline(root, responses, *, source="test", attempts=3):
    return GenerationPipeline(
        RecordedBackend(responses),
        provider="qwen",
        model="unit-test-model",
        output_dir=root,
        source=source,
        max_attempts=attempts,
    )


def test_ten_prompt_cases_have_counts_and_dimensions():
    cases = load_cases()
    assert len(cases) == len({case.id for case in cases}) == 10
    assert all(case.expectations.object_counts and case.expectations.map for case in cases)


def test_schema_valid_but_wrong_count_requires_repair(forest, tmp_path):
    wrong = forest.model_copy(deep=True)
    wrong.objects.append(
        wrong.objects[4].model_copy(update={"id": "extra_house", "position": [26, 12]})
    )
    pipeline = make_pipeline(tmp_path, [wrong.model_dump_json(), forest.model_dump_json()])
    result = pipeline.run(
        "Three houses", expectations=GenerationExpectations(object_counts={"house": 3})
    )
    assert result.success
    first = json.loads((result.directory / "attempts/001/validation.json").read_text())
    assert first["status"] == "invalid_expectations"
    assert "object_count" in pipeline.backend.calls[1][-1]["content"]
    assert load_world(result.directory / "world.json") == forest


def test_success_persists_prompt_model_attempts_png_and_validation(forest, tmp_path):
    pipeline = make_pipeline(tmp_path, ["not JSON", forest.model_dump_json()])
    result = pipeline.run("A forest village", case_id="forest")
    assert result.success
    record = json.loads((result.directory / "record.json").read_text())
    assert record["prompt"] == "A forest village"
    assert record["model"] == "unit-test-model"
    assert record["source"] == "test"
    assert len(record["attempts"]) == 2
    assert record["attempts"][1]["provider"]["request_id"] == "request-2"
    assert record["world_sha256"]
    assert record["finished_at"]
    assert (result.directory / "attempts/001/response.txt").read_text() == "not JSON"
    assert json.loads((result.directory / "validation.json").read_text())["passed"] is True
    with Image.open(result.directory / "map.png") as image:
        image.verify()


@pytest.mark.parametrize(
    "response,status",
    [
        ("invalid", "validation_failed"),
        (LLMError("HTTP 401", code="AuthenticationError"), "provider_error"),
        (
            LLMError("length", code="incomplete_response", raw_response='{"scene":'),
            "provider_error",
        ),
    ],
)
def test_failed_attempts_are_retained_without_fake_artifacts(response, status, tmp_path):
    result = make_pipeline(tmp_path, [response], attempts=1).run("A village")
    assert result.status == status
    record = json.loads((result.directory / "record.json").read_text())
    assert record["artifacts"] == {"world": None, "png": None}
    assert record["validation"]["passed"] is False
    assert len(record["attempts"]) == 1
    assert (result.directory / "attempts/001/request.json").exists()
    if isinstance(response, LLMError) and response.raw_response:
        assert (result.directory / "attempts/001/response.txt").read_text() == response.raw_response


def test_reruns_never_overwrite_previous_artifacts(forest, tmp_path):
    pipeline = make_pipeline(tmp_path, [forest.model_dump_json()] * 2)
    first = pipeline.run("Same prompt")
    before = (first.directory / "record.json").read_bytes()
    second = pipeline.run("Same prompt")
    assert first.run_id != second.run_id
    assert (first.directory / "record.json").read_bytes() == before


def test_render_failure_preserves_valid_world_and_error(forest, tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, [forest.model_dump_json()])

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(pipeline.renderer, "render", fail)
    result = pipeline.run("A village")
    assert result.status == "render_error"
    assert result.validation.passed
    assert (result.directory / "world.json").exists()
    assert not (result.directory / "map.png").exists()


def test_interrupt_leaves_an_auditable_record(tmp_path):
    pipeline = make_pipeline(tmp_path, [KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        pipeline.run("A village")
    record_path = next((tmp_path / "runs").glob("*/record.json"))
    assert json.loads(record_path.read_text())["status"] == "interrupted"


def test_spatial_failure_includes_numeric_footprints(forest):
    forest.objects[4].position = [31, 23]
    _, report = evaluate_response(forest.model_dump_json())
    assert report.status == "invalid_spatial"
    assert report.context["house_001"] == [31, 23, 34, 26]
    assert "Computed footprint bounds" in report.feedback()


def test_map_and_relation_expectations_are_checked(forest):
    expectations = GenerationExpectations(
        map={"width": 16, "height": 16, "biome": "snow"},
        minimum_relations={"inside": 2},
    )
    _, report = evaluate_response(forest.model_dump_json(), expectations)
    assert {item["code"] for item in report.errors} == {"map_mismatch", "relation_count"}


def test_feedback_only_suggests_removing_relations_not_required_by_case(forest):
    forest.objects[3].position = [17, 15]
    _, optional = evaluate_response(forest.model_dump_json(), GenerationExpectations())
    assert any("Remove the invalid optional connected_to" in hint for hint in optional.repair_hints)
    _, required = evaluate_response(
        forest.model_dump_json(), GenerationExpectations(minimum_relations={"connected_to": 2})
    )
    assert required.repair_hints == []
