import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from dataset_generation.spatial_curriculum import build
from dataset_generation.spatial_tasks import DIFFICULTIES, RELATIONS, check_task, make_task
from evaluation.benchmark import generate_benchmark, main
from evaluation.dataset_statistics import summarize
from evaluation.holdout import BENCHMARK_PATH, heldout_groups, reject_holdout
from evaluation.spatial import analyze
from training.data import read_jsonl
from world.evaluation import GenerationExpectations, evaluate_response


@pytest.mark.parametrize("relation", RELATIONS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_curriculum_tasks_are_feasible_seeded_and_targeted(relation, difficulty):
    row, world = make_task(11, relation, difficulty, 42)
    assert (row, world) == make_task(11, relation, difficulty, 42)
    _, report = evaluate_response(
        world.model_dump_json(), GenerationExpectations.model_validate(row["expectations"])
    )
    assert report.passed
    assert check_task(world, row) == []
    assert row["expectations"]["minimum_relations"][relation] >= 1
    assert row["focus_relation"] == relation and row["difficulty"] == difficulty
    assert not row["evaluation_only"]
    reject_holdout(row, heldout_groups())


def test_frozen_benchmark_distribution_reproducibility_and_no_answers(tmp_path):
    rows = generate_benchmark()
    assert rows == read_jsonl(BENCHMARK_PATH)
    frozen = json.loads(BENCHMARK_PATH.with_suffix(".manifest.json").read_text())
    assert frozen["sha256"] == hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest()
    assert len({row["prompt_group"] for row in rows}) == 100
    assert Counter(row["difficulty"] for row in rows) == {"easy": 30, "medium": 40, "hard": 30}
    assert all(row["evaluation_only"] and row["feasibility"]["validated"] for row in rows)
    assert all([m["role"] for m in row["messages"]] == ["system", "user"] for row in rows)
    output = tmp_path / "prompts.jsonl"
    main(["--output", str(output)])
    before = output.read_bytes()
    main(["--output", str(output)])
    assert output.read_bytes() == before
    with pytest.raises(ValueError, match="frozen"):
        main(["--output", str(output), "--seed", "9"])
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(before).hexdigest()


def test_holdout_cannot_be_bypassed_with_new_id_or_whitespace():
    row = read_jsonl(BENCHMARK_PATH)[0]
    disguised = {"id": "new_id", "prompt": "  " + row["prompt"].upper().replace(" ", "\n ")}
    with pytest.raises(ValueError, match="evaluation prompt"):
        reject_holdout(disguised, heldout_groups())
    with pytest.raises(ValueError, match="evaluation prompt"):
        reject_holdout({"evaluation_only": True}, set())


def test_targeted_builder_persists_real_pngs_and_honest_provenance(tmp_path):
    root = tmp_path / "data"
    result = build(root, num_samples=3, relation="inside", difficulty="medium")
    rows = read_jsonl(root / "train.jsonl")
    assert result["valid_count"] == 3 and result["failed_count"] == 0
    assert result["source_distribution"] == {"synthetic_constructive": 3}
    assert result["difficulty_distribution"] == {"medium": 3}
    assert result["focus_relation_distribution"] == {"inside": 3}
    assert not result["training_started"]
    for row in rows:
        assert row["model"] is None
        assert [item["role"] for item in row["messages"]] == ["system", "user", "assistant"]
        assert (root / row["artifacts"]["png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert row["validation"]["passed"]
    with pytest.raises(FileExistsError):
        build(root, num_samples=3)


def test_failed_candidates_are_excluded_and_counted(monkeypatch, tmp_path):
    from dataset_generation import spatial_curriculum

    original = spatial_curriculum.make_task
    monkeypatch.setattr(
        spatial_curriculum, "make_task", lambda *args: original(1, "near", "easy", 42)
    )
    report = build(tmp_path / "data", num_samples=2)
    assert report["valid_count"] == 1
    assert report["failed_count"] == 1
    assert report["failure_reason_distribution"] == {"duplicate_candidate": 1}
    assert len(read_jsonl(tmp_path / "data/failed.jsonl")) == 1


def test_benchmark_checks_sizes_ids_and_boundary_beyond_validator():
    row, world = make_task(85, "inside", "hard", 20260925, benchmark=True)
    row["messages"] = [
        {"role": "system", "content": "JSON"},
        {"role": "user", "content": row["prompt"]},
    ]
    predictions = [{"id": row["id"], "raw_response": world.model_dump_json()}]
    assert analyze([row], predictions)["metrics"]["task_constraint_pass_rate"] == 1
    row["required_objects"][0]["size"] = [1, 1]
    result = analyze([row], predictions)
    assert result["metrics"]["spatial_validator_pass_rate"] == 1
    assert result["metrics"]["task_constraint_pass_rate"] == 0
    assert result["details"][0]["requirement_failures"]


def test_statistics_counts_bad_records_as_failures():
    assert summarize([{"prompt": "broken", "messages": []}])["failed_count"] == 1


def test_training_admission_uses_frozen_holdout(monkeypatch, tmp_path):
    from training import prepare_dataset
    from training.config import load_config

    row = read_jsonl(BENCHMARK_PATH)[0] | {"status": "success"}
    monkeypatch.setattr(prepare_dataset, "source_rows", lambda root: iter([(row, root)]))
    called = []
    monkeypatch.setattr(prepare_dataset, "validate_record", lambda *args: called.append(args))
    config = load_config(Path(__file__).resolve().parents[1] / "training/configs/qwen7b_lora.json")
    config.data.max_samples = 1
    config.data.output_dir = str(tmp_path / "prepared")
    with pytest.raises(ValueError, match="only 0 available"):
        prepare_dataset.prepare(config)
    assert called == []
