import hashlib
import json
from collections import Counter
from copy import deepcopy
from typing import get_args

import pytest
from PIL import Image

from dataset_generation import dataset_builder as builder_module
from dataset_generation.dataset_builder import DatasetBuilder, collect_result, main
from dataset_generation.prompt_generator import (
    THEME_BIOMES,
    PromptSpec,
    generate_prompts,
    write_prompts,
)
from dataset_generation.qwen_generator import PacedBackend, QwenGenerator
from dataset_generation.storage import Checkpoints, output_lock
from gameworldlm.pipeline import GenerationPipeline
from llm.providers import LLMError, ProviderConfig
from world.evaluation import GenerationExpectations, evaluate_response
from world.schema import ObjectType


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_ten_thousand_unique_balanced_reproducible_prompts(tmp_path):
    cases = generate_prompts()
    assert (
        len(cases)
        == len({case.id for case in cases})
        == len({case.prompt for case in cases})
        == 10000
    )
    counts = Counter(case.theme for case in cases)
    assert set(counts) == set(THEME_BIOMES)
    assert max(counts.values()) - min(counts.values()) == 1
    assert len({case.descriptors["relation"] for case in cases}) == 4
    assert cases[:14] == generate_prompts(14)
    assert cases[:14] != generate_prompts(14, seed=43)
    for case in cases:
        assert case.expectations.map.biome == THEME_BIOMES[case.theme]
        assert set(case.expectations.object_counts) == set(get_args(ObjectType))
        assert 1 <= sum(case.expectations.object_counts.values()) <= 256
        assert len(case.prompt) <= 8000
    path = tmp_path / "prompts.jsonl"
    digest = write_prompts(path, cases)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert write_prompts(path, cases) == digest
    with pytest.raises(ValueError, match="inventory differs"):
        write_prompts(path, cases[:1])


@pytest.mark.parametrize("count", [0, -1, True, 1.2, 100001])
def test_invalid_inventory_size(count):
    with pytest.raises(ValueError):
        generate_prompts(count)


@pytest.fixture
def cases(forest, monkeypatch):
    counts = Counter(obj.object_type for obj in forest.objects)
    requests = [
        PromptSpec(
            id=f"forest_{index:03d}",
            theme="forest",
            prompt=f"Forest village variant {index}",
            expectations=GenerationExpectations(object_counts=dict(counts), map=forest.map),
            descriptors={"relation": "open"},
        )
        for index in range(5)
    ]
    monkeypatch.setattr(builder_module, "generate_prompts", lambda count, seed: requests[:count])
    return requests


class Backend:
    def __init__(self, responses, http_status=None):
        self.responses = iter(responses)
        self.http_status = http_status
        self.calls = []
        self.last_metadata = {}

    def complete(self, messages):
        self.calls.append(deepcopy(messages))
        self.last_metadata = {
            "actual_model": "test-qwen",
            "usage": {"total_tokens": 100},
            "http_status": self.http_status,
        }
        result = next(self.responses)
        if isinstance(result, BaseException):
            raise result
        return result


class TestGenerator:
    __test__ = False
    identity = {"provider": "qwen", "model": "test-qwen", "source": "llm"}

    def __init__(self, responses, http_status=None, attempts=1):
        self.backend = Backend(responses, http_status)
        self.attempts = attempts

    def generate(self, case, directory):
        pipeline = GenerationPipeline(
            self.backend,
            provider="qwen",
            model="test-qwen",
            source="llm",
            output_dir=directory,
            max_attempts=self.attempts,
            tile_size=16,
        )
        return pipeline.run(case.prompt, case_id=case.id, expectations=case.expectations)


def test_pipeline_outputs_stats_and_resume_without_duplicate_calls(cases, forest, tmp_path):
    generator = TestGenerator([forest.model_dump_json(), "invalid", forest.model_dump_json()])
    builder = DatasetBuilder(tmp_path, generator, count=3)
    first = builder.run(limit=2)
    assert first["processed"] == 2 and first["pending"] == 1
    assert first["success_rate"] == 0.5
    assert first["average_object_count"] == len(forest.objects)
    assert first["failure_reason_distribution"] == {"json_invalid": 1}
    assert first["reported_total_tokens"] == 200
    train = read_rows(tmp_path / "train.jsonl")
    failed = read_rows(tmp_path / "failed.jsonl")
    assert len(train) == len(failed) == 1
    assert "messages" not in failed[0]
    assert train[0]["messages"][1]["content"] == cases[0].prompt
    assert train[0]["messages"][-1]["role"] == "assistant"
    assert evaluate_response(train[0]["messages"][-1]["content"])[1].passed
    with Image.open(tmp_path / train[0]["artifacts"]["png"]) as image:
        image.verify()
    # A torn JSONL append is discarded and rebuilt from the committed checkpoint.
    with (tmp_path / "train.jsonl").open("a") as stream:
        stream.write('{"broken":')
    resumed = builder.run()
    assert resumed["status"] == "completed"
    assert resumed["succeeded"] == 2 and resumed["failed"] == 1
    assert resumed["pending"] == 0
    assert len(generator.backend.calls) == 3
    assert len(read_rows(tmp_path / "train.jsonl")) == 2
    builder.run()
    assert len(generator.backend.calls) == 3


def test_final_answer_excludes_repair_conversation(cases, forest, tmp_path):
    generator = TestGenerator(["invalid", forest.model_dump_json()], attempts=2)
    report = DatasetBuilder(tmp_path, generator, count=1).run()
    row = read_rows(tmp_path / "train.jsonl")[0]
    assert report["sdk_attempts"] == 2
    assert len(generator.backend.calls[-1]) == 4
    assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
    assert row["messages"][1]["content"] == cases[0].prompt


def test_recovery_after_pipeline_success_before_checkpoint(cases, forest, tmp_path, monkeypatch):
    generator = TestGenerator([forest.model_dump_json()] * 2)
    generate = generator.generate

    def crash(case, directory):
        generate(case, directory)
        raise RuntimeError("Simulated abrupt termination before commit")

    monkeypatch.setattr(generator, "generate", crash)
    with pytest.raises(RuntimeError):
        DatasetBuilder(tmp_path, generator, count=2).run()
    monkeypatch.setattr(generator, "generate", generate)
    report = DatasetBuilder(tmp_path, generator, count=2).run()
    assert report["succeeded"] == 2
    assert len(generator.backend.calls) == 2
    assert len(read_rows(tmp_path / "train.jsonl")) == 2


def test_interrupted_request_is_preserved_without_automatic_retry(cases, forest, tmp_path):
    generator = TestGenerator([KeyboardInterrupt(), forest.model_dump_json()])
    builder = DatasetBuilder(tmp_path, generator, count=2)
    report = builder.run()
    assert report["status"] == "interrupted"
    assert read_rows(tmp_path / "failed.jsonl")[0]["status"] == "interrupted"
    resumed = builder.run()
    assert resumed["failed"] == resumed["succeeded"] == 1
    assert len(generator.backend.calls) == 2


@pytest.mark.parametrize("http_status", [401, 403, 429])
def test_provider_circuit_breaker_preserves_pending_cases(cases, tmp_path, http_status):
    generator = TestGenerator([LLMError("test failure")], http_status=http_status)
    report = DatasetBuilder(tmp_path, generator, count=3).run()
    assert report["status"] == "paused" and report["pending"] == 2
    assert report["failed"] == 1
    assert str(http_status) in report["stop_reason"]
    assert len(generator.backend.calls) == 1


def test_repeated_transient_errors_stop_batch(cases, tmp_path):
    generator = TestGenerator([LLMError("network")] * 3)
    report = DatasetBuilder(tmp_path, generator, count=5).run()
    assert report["failed"] == 3 and report["pending"] == 2
    assert "Three consecutive" in report["stop_reason"]


def test_stop_file_and_token_threshold(cases, forest, tmp_path):
    generator = TestGenerator([forest.model_dump_json()] * 3)
    builder = DatasetBuilder(tmp_path, generator, count=3)
    (tmp_path / "STOP").touch()
    assert builder.run()["processed"] == 0
    (tmp_path / "STOP").unlink()
    report = builder.run(max_total_tokens=50)
    assert report["reported_total_tokens"] == 100 and report["pending"] == 2
    assert builder.run(max_total_tokens=50)["processed"] == 1
    assert len(generator.backend.calls) == 1


def test_changed_model_and_inventory_are_rejected(cases, forest, tmp_path):
    generator = TestGenerator([forest.model_dump_json()])
    DatasetBuilder(tmp_path, generator, count=2).run(limit=1)
    generator.identity = generator.identity | {"model": "another-model"}
    with pytest.raises(ValueError, match="configuration changed"):
        DatasetBuilder(tmp_path, generator, count=2).run()
    with pytest.raises(ValueError, match="inventory differs"):
        DatasetBuilder(tmp_path, generator, count=3).run()
    assert len(generator.backend.calls) == 1


def test_artifact_tampering_does_not_enter_training(cases, forest, tmp_path):
    generator = TestGenerator([forest.model_dump_json()])
    directory = tmp_path / "artifacts" / cases[0].id
    result = generator.generate(cases[0], directory)
    (result.directory / "world.json").write_text("{}")
    row = collect_result(cases[0], directory, tmp_path, generator.identity)
    assert row["status"] == "artifact_error"
    assert "messages" not in row


def test_render_failure_is_failed_and_stops_batch(cases, forest, tmp_path, monkeypatch):
    from engine.pygame_renderer import PygameRenderer

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(PygameRenderer, "render", fail)
    generator = TestGenerator([forest.model_dump_json()])
    report = DatasetBuilder(tmp_path, generator, count=2).run()
    row = read_rows(tmp_path / "failed.jsonl")[0]
    assert report["failed"] == 1 and report["pending"] == 1
    assert row["validation"]["passed"] is True
    assert row["status"] == "render_error"
    assert "world" in row["artifacts"] and "png" not in row["artifacts"]


def test_output_lock_is_exclusive_and_reusable(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(ValueError, match="Another dataset builder"):
            with output_lock(tmp_path):
                pass
    with output_lock(tmp_path):
        pass


def test_orphan_started_checkpoint_is_not_reissued(cases, forest, tmp_path):
    generator = TestGenerator([forest.model_dump_json()])
    DatasetBuilder(tmp_path, generator, count=2).run(limit=1)
    store = Checkpoints(tmp_path)
    store.start(cases[1].id)
    store.close()
    report = DatasetBuilder(tmp_path, generator, count=2).run()
    assert report["succeeded"] == report["failed"] == 1
    assert len(generator.backend.calls) == 1


def test_existing_artifacts_are_recovered_even_if_checkpoint_is_lost(cases, forest, tmp_path):
    generator = TestGenerator([forest.model_dump_json()])
    builder = DatasetBuilder(tmp_path, generator, count=1)
    builder.run()
    (tmp_path / "checkpoint.sqlite3").unlink()
    assert builder.run()["succeeded"] == 1
    assert len(generator.backend.calls) == 1


def test_interrupt_during_jsonl_append_recovers_committed_sample(
    cases, forest, tmp_path, monkeypatch
):
    generator = TestGenerator([forest.model_dump_json()])

    def interrupted_append(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(Checkpoints, "append_jsonl", interrupted_append)
    report = DatasetBuilder(tmp_path, generator, count=1).run()
    assert report["status"] == "interrupted"
    assert report["succeeded"] == 1
    assert len(read_rows(tmp_path / "train.jsonl")) == 1


def test_prepare_only_needs_no_credentials(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Offline preparation must not configure Qwen")

    monkeypatch.setattr(ProviderConfig, "from_env", unexpected)
    assert main(["--prepare-only", "--count", "14", "--output-dir", str(tmp_path)]) == 0
    assert len(read_rows(tmp_path / "prompts.jsonl")) == 14
    assert not (tmp_path / "train.jsonl").exists()


def test_qwen_adapter_and_repair_rate_limiting(cases, forest, tmp_path, monkeypatch):
    from dataset_generation import qwen_generator as qwen

    backend = Backend(["invalid", forest.model_dump_json()])
    backend.close = lambda: None
    monkeypatch.setattr(qwen, "OpenAICompatibleBackend", lambda config: backend)
    monkeypatch.setattr(qwen.time, "monotonic", lambda: 100.0)
    sleeps = []
    monkeypatch.setattr(qwen.time, "sleep", sleeps.append)
    config = ProviderConfig("qwen", "test-qwen", "unit-test-key", "https://example.invalid")
    generator = QwenGenerator(config, requests_per_minute=20)
    result = generator.generate(cases[0], tmp_path)
    assert result.success
    assert sleeps == [3.0]
    assert "api_key" not in generator.identity
    generator.close()


@pytest.mark.parametrize("rpm", [0, -1, float("inf"), float("nan")])
def test_invalid_rate_limit(rpm):
    with pytest.raises(ValueError):
        PacedBackend(Backend([]), rpm)
