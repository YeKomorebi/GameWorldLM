"""Strict raw-output metrics; never repair or extract JSON to improve model scores."""

import argparse
import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from dataset.storage import write_json
from world.evaluation import GenerationExpectations, evaluate_response
from world.schema import WorldState
from world.validator import WorldValidationError, validate_world

from .data import read_jsonl


def evaluate_predictions(samples: list[dict], predictions: list[dict]) -> dict:
    by_id = {}
    for row in predictions:
        if row["id"] in by_id:
            raise ValueError("Duplicate prediction ID")
        by_id[row["id"]] = row
    if len({row["id"] for row in samples}) != len(samples) or set(by_id) != {
        row["id"] for row in samples
    }:
        raise ValueError("Predictions must match the evaluation set exactly")
    details, totals = [], Counter()
    for sample in samples:
        prediction = by_id[sample["id"]]
        raw = prediction.get("raw_response", "")
        result = {
            "id": sample["id"],
            "json_parse": False,
            "schema_valid": False,
            "spatial_valid": False,
            "object_counts_correct": False,
            "expectations_pass": False,
            "error": prediction.get("error"),
        }
        if not result["error"]:
            try:
                json.loads(
                    raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))
                )
                result["json_parse"] = True
            except (ValueError, TypeError):
                pass
            if result["json_parse"]:
                try:
                    world = WorldState.model_validate_json(raw)
                    result["schema_valid"] = True
                    try:
                        validate_world(world)
                        result["spatial_valid"] = True
                    except WorldValidationError as exc:
                        result["spatial_errors"] = [issue.code for issue in exc.issues]
                    expected = sample["expectations"].get("object_counts", {})
                    if not expected:
                        target = WorldState.model_validate_json(sample["messages"][-1]["content"])
                        expected = Counter(obj.object_type for obj in target.objects)
                    actual = Counter(obj.object_type for obj in world.objects)
                    result["object_counts_correct"] = all(
                        actual[kind] == count for kind, count in expected.items()
                    )
                    result["expected_counts"], result["actual_counts"] = (
                        dict(expected),
                        dict(actual),
                    )
                    _, report = evaluate_response(
                        raw, GenerationExpectations.model_validate(sample["expectations"])
                    )
                    result["expectations_pass"] = report.passed
                except ValidationError as exc:
                    result["schema_errors"] = [
                        error["type"] for error in exc.errors(include_input=False)
                    ]
        totals.update(
            {
                name: int(result[name])
                for name in (
                    "json_parse",
                    "schema_valid",
                    "spatial_valid",
                    "object_counts_correct",
                    "expectations_pass",
                )
            }
        )
        details.append(result)
    count = len(samples)
    return {
        "samples": count,
        "counts": dict(totals),
        "metrics": {
            "json_parse_rate": totals["json_parse"] / count if count else None,
            "schema_valid_rate": totals["schema_valid"] / count if count else None,
            "spatial_validator_pass_rate": totals["spatial_valid"] / count if count else None,
            "object_count_accuracy": totals["object_counts_correct"] / count if count else None,
            "all_expectations_pass_rate": totals["expectations_pass"] / count if count else None,
        },
        "details": details,
        "definitions": {
            "denominator": (
                "all samples, including invalid JSON, schema failures and inference errors"
            ),
            "object_count_accuracy": "fraction with every explicitly expected type count correct",
            "parsing": "strict raw JSON, no code-fence stripping, repair or substring extraction",
        },
    }


def comparison_report(root: Path, base: dict, lora: dict, metadata: dict) -> Path:
    if base["samples"] != lora["samples"] or [r["id"] for r in base["details"]] != [
        r["id"] for r in lora["details"]
    ]:
        raise ValueError("Base and LoRA evaluation sets differ")
    lines = [
        "# GameWorldLM Base vs LoRA",
        "",
        f"Evaluation samples: {base['samples']}",
        "",
        f"Base model: `{metadata['model']}`",
        f"Mode: `{metadata['mode']}`",
        "",
        "| Metric | Base | LoRA | Change (percentage points) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, value in base["metrics"].items():
        improved = lora["metrics"][name]
        lines.append(f"| {name} | {value:.2%} | {improved:.2%} | {(improved - value) * 100:+.2f} |")
    lines.extend(
        [
            "",
            "Both variants use the same prompts, precision, tokenizer, "
            "decoding parameters and seed.",
            "The frozen base is evaluated with the adapter disabled. "
            "No validation repair is applied.",
            "All rates include malformed responses in the denominator. "
            "This small dataset is a pipeline check, not a performance claim.",
            "",
            "## Run Metadata",
            "",
            "```json",
            json.dumps(metadata, indent=2),
            "```",
            "",
        ]
    )
    path = root / "base_vs_lora_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate stored base and LoRA predictions")
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--lora", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    samples = read_jsonl(args.samples)
    base = evaluate_predictions(samples, read_jsonl(args.base))
    lora = evaluate_predictions(samples, read_jsonl(args.lora))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "base_metrics.json", base)
    write_json(args.output_dir / "lora_metrics.json", lora)
    path = comparison_report(
        args.output_dir,
        base,
        lora,
        {"model": "see prediction metadata", "mode": "offline evaluation"},
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
