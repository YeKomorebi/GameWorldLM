import hashlib
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime

import pytest

from dataset_generation import dataset_builder as builder_module
from dataset_generation import splits
from dataset_generation.dataset_builder import DatasetBuilder
from dataset_generation.prompt_generator import (
    PromptSpec,
    generate_prompts,
    load_prompts,
    write_prompts,
)
from dataset_generation.splits import export_records, export_snapshot, split_sizes
from dataset_generation.statistics import Statistics, generation_duration
from dataset_generation.storage import Checkpoints, output_lock, read_snapshot
from llm.prompt_parser import build_messages
from llm.providers import ProviderConfig
from tests.test_dataset_generation import TestGenerator, read_rows
from world.evaluation import GenerationExpectations


def case_for_world(index, world):
    return PromptSpec(
        id=f"scene_{index:04d}",
        theme="forest",
        prompt=f"Forest training scene {index}",
        expectations=GenerationExpectations(
            object_counts=dict(Counter(obj.object_type for obj in world.objects)),
            map=world.map,
        ),
        descriptors={"relation": "open"},
    )


@pytest.fixture
def completed(tmp_path, forest):
    source = tmp_path / "source"
    cases = [case_for_world(i, forest) for i in range(10)]
    generator = TestGenerator([forest.model_dump_json()] * len(cases))
    builder = DatasetBuilder(source, generator, cases=cases)
    report = builder.run()
    return source, builder, generator, report


def test_existing_prompt_subset_preserves_ids_text_and_expectations(tmp_path):
    cases = generate_prompts(14)
    path = tmp_path / "existing.jsonl"
    write_prompts(path, cases)
    before = path.read_bytes()
    assert load_prompts(path, 7) == cases[:7]
    assert load_prompts(path) == cases
    assert path.read_bytes() == before
    array_path = tmp_path / "existing.json"
    array_path.write_text(json.dumps([case.model_dump() for case in cases]), encoding="utf-8")
    assert load_prompts(array_path, 3) == cases[:3]
    with pytest.raises(ValueError, match="only 14"):
        load_prompts(path, 15)


@pytest.mark.parametrize("number", [0, -1, True, 1.5])
def test_invalid_selection_sizes(tmp_path, number):
    path = tmp_path / "prompts.jsonl"
    write_prompts(path, generate_prompts(2))
    with pytest.raises(ValueError):
        load_prompts(path, number)
    with pytest.raises(ValueError):
        read_snapshot(tmp_path, number)


@pytest.mark.parametrize("problem", ["id", "prompt", "blank", "invalid_json", "empty"])
def test_invalid_existing_inventory_fails_before_generation(tmp_path, problem):
    cases = generate_prompts(2)
    if problem == "id":
        cases[1].id = cases[0].id
    elif problem == "prompt":
        cases[1].prompt = "  " + cases[0].prompt.upper() + "  "
    elif problem == "blank":
        cases[1].prompt = "   "
    path = tmp_path / "prompts.jsonl"
    write_prompts(path, cases)
    if problem == "invalid_json":
        path.write_text("{\n", encoding="utf-8")
    elif problem == "empty":
        path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompts(path)


def test_cli_num_samples_selects_existing_prompts_without_credentials(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    cases = generate_prompts(14)
    write_prompts(source, cases)
    target = tmp_path / "selected"

    def unexpected(*args, **kwargs):
        raise AssertionError("Offline selection must not configure an API client")

    monkeypatch.setattr(builder_module, "QwenGenerator", unexpected)
    assert (
        builder_module.main(
            [
                "--prompts-file",
                str(source),
                "--num-samples",
                "7",
                "--prepare-only",
                "--output-dir",
                str(target),
            ]
        )
        == 0
    )
    assert load_prompts(target / "prompts.jsonl") == cases[:7]
    assert len(load_prompts(source)) == 14


def test_default_ten_thousand_and_legacy_count_alias(tmp_path, monkeypatch):
    calls = []

    def capture(count, seed):
        calls.append(count)
        return generate_prompts(2)

    monkeypatch.setattr(builder_module, "generate_prompts", capture)
    for index, arguments in enumerate(([], ["--count", "1000"], ["--num-samples", "1000"])):
        assert (
            builder_module.main(
                [
                    "--prepare-only",
                    "--output-dir",
                    str(tmp_path / str(index)),
                    *arguments,
                ]
            )
            == 0
        )
    assert calls == [10000, 1000, 1000]


@pytest.mark.parametrize(
    "number,expected",
    [
        (0, (0, 0, 0)),
        (1, (1, 0, 0)),
        (2, (2, 0, 0)),
        (7, (5, 1, 1)),
        (10, (8, 1, 1)),
        (11, (9, 1, 1)),
        (1000, (800, 100, 100)),
        (10000, (8000, 1000, 1000)),
    ],
)
def test_largest_remainder_split_sizes(number, expected):
    assert tuple(split_sizes(number).values()) == expected
    assert sum(split_sizes(number).values()) == number


def test_automatic_split_statistics_and_reproducible_resume(completed, forest):
    source, builder, generator, report = completed
    assert report["splits"] == {"train": 8, "val": 1, "test": 1}
    assert report["total_count"] == report["valid_count"] == 10
    assert report["failed_count"] == 0
    assert len(read_rows(source / "valid.jsonl")) == 10
    counts = Counter(obj.object_type for obj in forest.objects)
    for kind, count in counts.items():
        assert report["object_distribution"][kind] == count * 10
    assert sum(report["object_distribution"].values()) == 10 * len(forest.objects)
    membership = {}
    files = {}
    for name, count in report["splits"].items():
        path = source / f"{name}.jsonl"
        files[name] = path.read_bytes()
        rows = read_rows(path)
        assert len(rows) == count
        for row in rows:
            assert row["id"] not in membership
            membership[row["id"]] = name
    assert len(membership) == 10
    builder.run()
    assert len(generator.backend.calls) == 10
    assert all(
        (source / f"{name}.jsonl").read_bytes() == content for name, content in files.items()
    )
    manifest = json.loads((source / "split_manifest.json").read_text())
    for filename, checksum in manifest["files_sha256"].items():
        assert hashlib.sha256((source / filename).read_bytes()).hexdigest() == checksum
    mapping = json.loads((source / "dataset_info.json").read_text())
    assert {entry["file_name"] for entry in mapping.values()} == {
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
    }


def test_failure_exclusion_and_split_over_valid_samples(tmp_path, forest):
    cases = [case_for_world(i, forest) for i in range(11)]
    generator = TestGenerator([forest.model_dump_json()] * 10 + ["invalid JSON"])
    report = DatasetBuilder(tmp_path, generator, cases=cases).run()
    assert report["total_count"] == 11
    assert report["valid_count"] == 10 and report["failed_count"] == 1
    assert report["splits"] == {"train": 8, "val": 1, "test": 1}
    assert sum(report["object_distribution"].values()) == 10 * len(forest.objects)
    failed = read_rows(tmp_path / "failed.jsonl")
    assert len(failed) == 1 and "messages" not in failed[0]


def test_snapshot_reads_live_source_without_mutation(completed, tmp_path):
    source, _, generator, _ = completed
    before = (source / "checkpoint.sqlite3").read_bytes()
    source_train = (source / "train.jsonl").read_bytes()
    with output_lock(source):
        report = export_snapshot(source, tmp_path / "snapshot", num_samples=10)
    assert report["splits"] == {"train": 8, "val": 1, "test": 1}
    assert (source / "checkpoint.sqlite3").read_bytes() == before
    assert (source / "train.jsonl").read_bytes() == source_train
    assert len(generator.backend.calls) == 10
    sample = read_rows(tmp_path / "snapshot/train.jsonl")[0]
    world = tmp_path / "snapshot" / sample["artifact_root"] / sample["artifacts"]["world"]
    assert world.is_file()


def test_snapshot_excludes_inflight_records_and_refuses_short_selection(completed, tmp_path):
    source, _, _, _ = completed
    store = Checkpoints(source)
    store.start("inflight")
    store.close()
    assert len(read_snapshot(source)) == 10
    with pytest.raises(ValueError, match="only 10"):
        export_snapshot(source, tmp_path / "too_many", num_samples=11)
    assert not (tmp_path / "too_many").exists()
    with pytest.raises(ValueError, match="new directory"):
        export_snapshot(source, source)


def test_revalidation_rejects_corrupted_sample(completed, tmp_path):
    source, _, _, _ = completed
    row = read_snapshot(source, 1)[0]
    (source / row["artifacts"]["world"]).write_text("{}")
    report = export_snapshot(source, tmp_path / "checked")
    assert report["valid_count"] == 9 and report["failed_count"] == 1
    assert sum(report["splits"].values()) == 9
    failed = read_rows(tmp_path / "checked/failed.jsonl")
    assert failed[0]["status"] == "export_rejected"
    assert "messages" not in failed[0]


def test_thousand_prevalidated_records_are_complete_disjoint_and_seeded(
    completed, tmp_path, monkeypatch
):
    source, _, _, _ = completed
    base = read_snapshot(source, 1)[0]
    records = []
    for index in range(1000):
        row = deepcopy(base)
        row["id"] = f"sample_{index:04d}"
        row["prompt"] = f"Distinct instruction {index}"
        row["messages"] = build_messages(row["prompt"]) + [row["messages"][-1]]
        records.append(row)
    # Isolate partitioning from artifact verification, which is tested end to end above.
    monkeypatch.setattr(splits, "_validate_success", lambda row, source: row)
    destinations = [tmp_path / name for name in ("first", "reordered", "other_seed")]
    for path, rows, seed in zip(
        destinations, (records, list(reversed(records)), records), (42, 42, 43), strict=True
    ):
        path.mkdir()
        report = export_records(rows, path, source_root=source, seed=seed)
        assert report["splits"] == {"train": 800, "val": 100, "test": 100}
        memberships = [read_rows(path / f"{name}.jsonl") for name in ("train", "val", "test")]
        assert len({row["id"] for rows in memberships for row in rows}) == 1000
    assignments = [
        {row["id"]: row["split"] for row in read_rows(path / "split_membership.jsonl")}
        for path in destinations
    ]
    assert assignments[0] == assignments[1]
    assert assignments[0] != assignments[2]


def test_duplicate_normalized_prompts_cannot_leak_across_splits(completed, tmp_path, monkeypatch):
    source, _, _, _ = completed
    rows = read_snapshot(source, 2)
    rows[1]["prompt"] = "  " + rows[0]["prompt"].upper() + "  "
    monkeypatch.setattr(splits, "_validate_success", lambda row, source: row)
    destination = tmp_path / "dedup"
    destination.mkdir()
    report = export_records(rows, destination, source_root=source)
    assert report["valid_count"] == 1 and report["failed_count"] == 1
    assert "Duplicate" in read_rows(destination / "failed.jsonl")[0]["error"]


def test_estimated_completion_uses_actual_duration_and_handles_pauses():
    stats = Statistics(100)
    for duration in (20, 40, 60):
        stats.add(
            {
                "status": "success",
                "theme": "forest",
                "object_counts": {"tree": 2},
                "attempts": 1,
                "reported_total_tokens": 10,
                "attempts_without_usage": 0,
                "failure_reasons": [],
                "generation_seconds": duration,
            }
        )
    report = stats.report("running")
    assert report["average_generation_seconds"] == 40
    assert report["estimated_remaining_seconds"] == 3880
    eta = datetime.fromisoformat(report["estimated_completion_at"])
    now = datetime.fromisoformat(report["updated_at"])
    assert (eta - now).total_seconds() == 3880
    assert stats.report("paused")["estimated_completion_at"] is None
    assert stats.report("paused")["estimated_remaining_seconds"] == 3880
    assert Statistics(100).report("running")["estimated_completion_at"] is None
    assert generation_duration({}) is None
    assert generation_duration({"started_at": "2026-01-01", "finished_at": "2025-01-01"}) is None


def test_cli_existing_prompts_generation_to_training_splits(tmp_path, forest, monkeypatch):
    cases = [case_for_world(i, forest) for i in range(14)]
    source = tmp_path / "input.jsonl"
    write_prompts(source, cases)
    generator = TestGenerator([forest.model_dump_json()] * 10)
    closed = []
    generator.close = lambda: closed.append(True)
    config = ProviderConfig("qwen", "test-qwen", "test-key", "https://example.invalid")
    monkeypatch.setattr(builder_module, "QwenGenerator", lambda *args, **kwargs: generator)
    monkeypatch.setattr(builder_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(ProviderConfig, "from_env", lambda *args, **kwargs: config)
    target = tmp_path / "output"
    assert (
        builder_module.main(
            [
                "--prompts-file",
                str(source),
                "--num-samples",
                "10",
                "--output-dir",
                str(target),
            ]
        )
        == 0
    )
    assert len(generator.backend.calls) == 10
    assert load_prompts(target / "prompts.jsonl") == cases[:10]
    assert json.loads((target / "statistics.json").read_text())["splits"] == {
        "train": 8,
        "val": 1,
        "test": 1,
    }
    assert closed == [True]
