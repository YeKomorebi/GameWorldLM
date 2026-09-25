import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataset.storage import schema_digest
from tests.test_dataset_generation import TestGenerator
from tests.test_dataset_splits import case_for_world
from training.config import load_config
from training.data import (
    AssistantCollator,
    assistant_windows,
    encode_chat,
    read_jsonl,
    tokenize_rows,
)
from training.evaluate_lora import comparison_report, evaluate_predictions
from training.prepare_dataset import partition_counts, prepare, verify_prepared
from training.train_lora import check_run_directory

CONFIG = Path(__file__).resolve().parents[1] / "training/configs/qwen7b_lora.json"


def test_configuration_defaults_and_qlora_smoke_overrides():
    config = load_config(CONFIG)
    assert config.model.name_or_path == "Qwen/Qwen2.5-7B-Instruct"
    assert (config.lora.r, config.lora.alpha, config.lora.dropout) == (16, 32, 0.05)
    assert config.training.epochs == 3
    assert config.training.learning_rate == 2e-4
    assert config.training.max_seq_length == 2048
    assert not config.model.load_in_4bit
    assert load_config(CONFIG.with_name("qwen7b_qlora.json")).model.load_in_4bit
    smoke = load_config(CONFIG.with_name("overfit_smoke.json"))
    assert smoke.smoke.enabled and smoke.smoke.num_samples == 20
    assert smoke.training.output_dir != config.training.output_dir
    assert smoke.training.epochs > config.training.epochs
    assert partition_counts(143, config.data.split_ratios) == [115, 14, 14]


def test_config_rejects_unknown_fields_and_inheritance_cycles(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"extends": str(CONFIG), "training": {"epohs": 2}}))
    with pytest.raises(ValueError):
        load_config(path)
    path.write_text(json.dumps({"extends": "bad.json"}))
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(path)


def test_resume_rejects_changed_configuration_or_dataset(tmp_path):
    config = load_config(CONFIG)
    config.project_root = str(tmp_path)
    output = config.path(config.training.output_dir)
    checkpoint = output / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text('{"global_step":10}')
    (output / "resolved_config.json").write_text(config.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        check_run_directory(config)
    config.training.resume_from_checkpoint = str(checkpoint)
    prepared = config.path(config.data.output_dir)
    prepared.mkdir(parents=True)
    manifest = {"dataset_sha256": "current", "schema_sha256": schema_digest(), "files_sha256": {}}
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    (output / "dataset_manifest.json").write_text(json.dumps(manifest))
    check_run_directory(config)
    config.training.learning_rate *= 2
    with pytest.raises(ValueError, match="configuration differs"):
        check_run_directory(config)
    config.training.learning_rate /= 2
    (output / "dataset_manifest.json").write_text('{"dataset_sha256":"different"}')
    with pytest.raises(ValueError, match="dataset differs"):
        check_run_directory(config)


@pytest.mark.parametrize(
    "length,boundary", [(40, 10), (300, 90), (800, 250), (500, 128), (700, 512)]
)
def test_assistant_only_lossless_window_coverage(length, boundary):
    tokens = list(range(length))
    windows = assistant_windows(
        tokens, boundary, max_length=128, overlap=32, policy="chunk_assistant"
    )
    supervised = []
    for row in windows:
        assert len(row["input_ids"]) <= 128
        assert len(row["labels"]) == len(row["input_ids"]) == len(row["attention_mask"])
        assert row["labels"][0] == -100
        supervised.extend(label for label in row["labels"] if label != -100)
        for token, label in zip(row["input_ids"], row["labels"], strict=True):
            if token < boundary:
                assert label == -100
            if label != -100:
                assert label == token
    assert supervised == tokens[boundary:]


def test_long_sequence_never_silently_truncates():
    with pytest.raises(ValueError, match="no truncation"):
        assistant_windows(list(range(300)), 50, 128, 32, "error")


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize
        return list(range(12 if add_generation_prompt else 25))


def test_chat_template_prefix_and_tokenization_stats():
    messages = [{"role": role, "content": role} for role in ("system", "user", "assistant")]
    tokens, start = encode_chat(FakeTokenizer(), messages)
    assert start == 12 and len(tokens) == 25
    rows, report = tokenize_rows(
        FakeTokenizer(),
        [{"messages": messages}],
        SimpleNamespace(
            max_seq_length=16,
            overlap_tokens=4,
            long_sequence_policy="chunk_assistant",
        ),
    )
    assert report["long_samples"] == 1 and report["truncated_tokens"] == 0
    assert report["supervised_tokens"] == 13
    assert all(row["labels"][0] == -100 for row in rows)


def test_ambiguous_template_is_rejected():
    class BrokenTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return [999] if add_generation_prompt else list(range(10))

    with pytest.raises(ValueError, match="unambiguous"):
        encode_chat(BrokenTokenizer(), [{"role": role} for role in ("system", "user", "assistant")])


def test_padding_and_prompt_labels_are_ignored_even_when_pad_equals_eos():
    pytest.importorskip("torch")
    collator = AssistantCollator(pad_token_id=9, multiple=4)
    rows = [
        {"input_ids": [1, 2, 9], "attention_mask": [1, 1, 1], "labels": [-100, 2, 9]},
        {"input_ids": [3, 9], "attention_mask": [1, 1], "labels": [-100, 9]},
    ]
    batch = collator(rows)
    assert batch["labels"].tolist() == [[-100, 2, 9, -100], [-100, 9, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]


def test_trainer_counts_partial_accumulation_as_a_full_epoch(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    trainer = transformers.Trainer(
        model=torch.nn.Linear(1, 1),
        args=transformers.TrainingArguments(
            output_dir=str(tmp_path),
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            use_cpu=True,
            report_to=[],
        ),
        train_dataset=[{"input_ids": [1]} for _ in range(115)],
    )
    values = trainer.set_initial_training_values(trainer.args, trainer.get_train_dataloader(), 4)
    assert values[0] == 3
    assert values[1] == 29
    assert values[-1] == 87


def sample(world, sample_id="example"):
    return {
        "id": sample_id,
        "expectations": {"object_counts": {"house": 3}, "map": None, "minimum_relations": {}},
        "messages": [
            {"role": "system", "content": "JSON"},
            {"role": "user", "content": "village"},
            {"role": "assistant", "content": world.model_dump_json()},
        ],
    }


def test_evaluation_separates_json_schema_spatial_and_counts(forest):
    samples = [sample(forest, str(i)) for i in range(5)]
    outside = forest.model_copy(deep=True)
    outside.objects[4].position = [127, 127]
    wrong_count = forest.model_copy(deep=True)
    wrong_count.objects[4].object_type = "ruin"
    predictions = [
        {"id": str(i), "raw_response": raw}
        for i, raw in enumerate(
            [
                "not JSON",
                "{}",
                outside.model_dump_json(),
                wrong_count.model_dump_json(),
                forest.model_dump_json(),
            ]
        )
    ]
    metrics = evaluate_predictions(samples, predictions)["metrics"]
    assert metrics["json_parse_rate"] == 4 / 5
    assert metrics["schema_valid_rate"] == 3 / 5
    assert metrics["spatial_validator_pass_rate"] == 2 / 5
    assert metrics["object_count_accuracy"] == 2 / 5


def test_markdown_invalid_json_and_missing_prediction_not_silently_fixed(forest):
    rows = [sample(forest)]
    result = evaluate_predictions(
        rows, [{"id": "example", "raw_response": f"```json\n{forest.model_dump_json()}\n```"}]
    )
    assert result["metrics"]["json_parse_rate"] == 0
    with pytest.raises(ValueError, match="exactly"):
        evaluate_predictions(rows, [])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_predictions(rows, [{"id": "example"}] * 2)


def test_report_compares_same_ids_and_denominators(forest, tmp_path):
    rows = [sample(forest)]
    base = evaluate_predictions(rows, [{"id": "example", "raw_response": "{}"}])
    lora = evaluate_predictions(rows, [{"id": "example", "raw_response": forest.model_dump_json()}])
    path = comparison_report(tmp_path, base, lora, {"model": "test", "mode": "held-out test"})
    assert "100.00%" in path.read_text() and "+100.00" in path.read_text()
    bad = deepcopy(lora)
    bad["details"][0]["id"] = "different"
    with pytest.raises(ValueError, match="differ"):
        comparison_report(tmp_path, base, bad, {"model": "test", "mode": "test"})


def test_prepare_validates_history_and_freezes_chat_splits(forest, tmp_path):
    config = load_config(CONFIG)
    config.project_root = str(tmp_path)
    config.data.sources = ["source"]
    config.data.output_dir = "prepared"
    config.data.max_samples = 10
    config.data.system_prompt_file = str(CONFIG.with_name("system_prompt.txt"))
    generator = TestGenerator([forest.model_dump_json()] * 11)
    source = tmp_path / "source"
    # Legacy pipeline directories have original system messages which may predate current prompts.
    for index in range(11):
        generator.generate(case_for_world(index, forest), source)
    report = prepare(config)
    assert report["available_unique_valid"] == 11 and report["selected"] == 10
    assert report["splits"] == {"train": 8, "val": 1, "test": 1}
    assert report["schema_sha256"] == schema_digest()
    rows = [
        row
        for name in ("train", "val", "test")
        for row in read_jsonl(tmp_path / f"prepared/{name}.jsonl")
    ]
    assert len({row["prompt_group"] for row in rows}) == 10
    assert all(
        [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
        for row in rows
    )
    assert all(json.loads(row["messages"][-1]["content"])["objects"] for row in rows)
    assert prepare(config) == report
    path = tmp_path / "prepared/train.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        verify_prepared(tmp_path / "prepared")


def test_object_counts_ignore_invalid_schema_and_inference_errors(forest):
    rows = [sample(forest, "a"), sample(forest, "b")]
    result = evaluate_predictions(
        rows,
        [
            {"id": "a", "raw_response": forest.model_dump_json(), "error": "generation failed"},
            {"id": "b", "raw_response": "NaN"},
        ],
    )
    assert result["metrics"]["json_parse_rate"] == 0
    assert result["metrics"]["object_count_accuracy"] == 0
