import hashlib
import json

import pytest

from dataset.export import export_dataset
from dataset.storage import write_json
from tests.test_real_pipeline import make_pipeline


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_exports_clean_sft_and_alpaca_with_original_prompt_after_repair(forest, tmp_path):
    pipeline = make_pipeline(tmp_path, ["{}", forest.model_dump_json()], source="llm")
    result = pipeline.run("Three houses")
    report = export_dataset(tmp_path, tmp_path / "dataset")
    assert report["accepted"] == 1
    sample = read_jsonl(tmp_path / "dataset/train.jsonl")[0]
    assert [m["role"] for m in sample["messages"]] == ["system", "user", "assistant"]
    assert sample["messages"][1]["content"] == "Three houses"
    assert json.loads(sample["messages"][2]["content"]) == forest.model_dump()
    alpaca = read_jsonl(tmp_path / "dataset/alpaca_train.jsonl")[0]
    assert alpaca["instruction"] == "Three houses"
    assert alpaca["output"] == sample["messages"][2]["content"]
    records = read_jsonl(tmp_path / "dataset/records.jsonl")
    assert records[0]["run_id"] == result.run_id
    assert len(records[0]["attempts"]) == 2


def test_excludes_failures_and_fixture_provenance(forest, tmp_path):
    make_pipeline(tmp_path, ["{}"], source="llm", attempts=1).run("Bad world")
    make_pipeline(tmp_path, [forest.model_dump_json()], source="fixture").run("Fixture")
    report = export_dataset(tmp_path, tmp_path / "dataset")
    assert report["accepted"] == 0
    assert len(report["skipped"]) == 2
    assert len(read_jsonl(tmp_path / "dataset/records.jsonl")) == 2
    assert read_jsonl(tmp_path / "dataset/train.jsonl") == []


def test_duplicate_completions_removed_and_prompt_groups_never_cross_splits(forest, tmp_path):
    for i in range(5):
        make_pipeline(tmp_path, [forest.model_dump_json()], source="llm").run(f"Prompt {i}")
    make_pipeline(tmp_path, [forest.model_dump_json()], source="llm").run("  PROMPT   0 ")
    variation = forest.model_copy(deep=True)
    variation.scene = "different_scene"
    make_pipeline(tmp_path, [variation.model_dump_json()], source="llm").run("Prompt 0")
    report = export_dataset(tmp_path, tmp_path / "dataset")
    assert report["accepted"] == 6
    assert report["duplicates_removed"] == 1
    members = read_jsonl(tmp_path / "dataset/membership.jsonl")
    groups = {}
    for item in members:
        groups.setdefault(item["prompt_group"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    assert report["splits"]["validation"] > 0
    export_dataset(tmp_path, tmp_path / "again")
    assert (tmp_path / "again/train.jsonl").read_bytes() == (
        tmp_path / "dataset/train.jsonl"
    ).read_bytes()


@pytest.mark.parametrize("tamper", ["hash", "invalid_spatial", "png", "request"])
def test_tampered_or_incomplete_artifacts_are_excluded(forest, tmp_path, tamper):
    result = make_pipeline(tmp_path, [forest.model_dump_json()], source="llm").run("A village")
    world_path = result.directory / "world.json"
    if tamper == "hash":
        world_path.write_text("{}")
    elif tamper == "invalid_spatial":
        forest.objects[4].position = [31, 23]
        world_path.write_text(forest.model_dump_json(), encoding="utf-8")
        record_path = result.directory / "record.json"
        record = json.loads(record_path.read_text())
        record["world_sha256"] = hashlib.sha256(world_path.read_bytes()).hexdigest()
        write_json(record_path, record)
    elif tamper == "png":
        (result.directory / "map.png").unlink()
    else:
        write_json(result.directory / "attempts/001/request.json", {"messages": []})
    report = export_dataset(tmp_path, tmp_path / "dataset")
    assert report["accepted"] == 0
    assert len(report["skipped"]) == 1


def test_existing_dataset_directory_is_not_overwritten(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "dataset").mkdir()
    with pytest.raises(FileExistsError):
        export_dataset(tmp_path, tmp_path / "dataset")
