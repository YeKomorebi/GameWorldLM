"""Unrepaired prediction analysis using the canonical validator and geometry."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from dataset.storage import write_json
from training.data import read_jsonl
from training.evaluate_lora import evaluate_predictions
from world.geometry import bounds, contains, shares_edge, tile_distance
from world.schema import WorldState
from world.validator import WorldValidationError, validate_world

RELATIONS = ("near", "inside", "connected_to")
CATEGORIES = (
    "overlap",
    "out_of_bounds",
    "near_violation",
    "inside_violation",
    "connected_to_violation",
    "invalid_reference",
    "other",
)
CODE_CATEGORY = {
    "overlap": "overlap",
    "blocked_entity": "overlap",
    "out_of_bounds": "out_of_bounds",
    "not_near": "near_violation",
    "not_inside": "inside_violation",
    "not_connected": "connected_to_violation",
    "missing_reference": "invalid_reference",
}


def relation_key(relation) -> tuple:
    item = relation.model_dump() if hasattr(relation, "model_dump") else relation
    pair = (item["source"], item["target"])
    if item["relation_type"] != "inside":
        pair = tuple(sorted(pair))
    return (item["relation_type"], *pair)


def relation_scores(world: WorldState | None, sample: dict) -> dict:
    """Micro slots penalize absent requirements, duplicate edges and invalid references."""
    required = sample.get("required_relations", [])
    minimums = sample.get("expectations", {}).get("minimum_relations", {})
    by_id = {obj.id: obj for obj in world.objects} if world else {}
    id_counts = Counter(obj.id for obj in world.objects) if world else Counter()
    scores = {}
    for kind in RELATIONS:
        predicted = [rel for rel in world.relations if rel.relation_type == kind] if world else []
        wanted = {relation_key(rel) for rel in required if rel["relation_type"] == kind}
        seen, valid = set(), set()
        checks = []
        for rel in predicted:
            key = relation_key(rel)
            reason = None
            if key in seen:
                reason = "duplicate_relation"
            elif rel.source == rel.target:
                reason = "self_relation"
            elif id_counts[rel.source] != 1 or id_counts[rel.target] != 1:
                reason = "missing_or_ambiguous_reference"
            else:
                source, target = by_id[rel.source], by_id[rel.target]
                passes = {
                    "near": tile_distance(source, target) <= 3,
                    "inside": contains(target, source) and bounds(target) != bounds(source),
                    "connected_to": shares_edge(source, target),
                }[kind]
                if not passes:
                    reason = {
                        "near": "not_near",
                        "inside": "not_inside",
                        "connected_to": "not_connected",
                    }[kind]
            if reason is None:
                valid.add(key)
            seen.add(key)
            checks.append(rel.model_dump() | {"valid": reason is None, "reason": reason})
        # An unrelated valid edge must not replace an explicitly requested edge.
        missing = wanted - seen
        denominator = max(len(predicted) + len(missing), minimums.get(kind, 0), len(wanted))
        scores[kind] = {
            "passed": len(valid),
            "total": denominator,
            "accuracy": len(valid) / denominator if denominator else None,
            "required_minimum": minimums.get(kind, 0),
            "missing_required_edges": [list(key) for key in sorted(missing)],
            "missing_slots": denominator - len(predicted),
            "checks": checks,
        }
    return scores


def _strict_json(raw: str):
    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(raw, parse_constant=reject_constant)


def analyze(samples: list[dict], predictions: list[dict]) -> dict:
    legacy = evaluate_predictions(samples, predictions)
    by_id = {row["id"]: row for row in predictions}
    details, category_issues, category_samples, raw_codes = [], Counter(), Counter(), Counter()
    relation_totals = {kind: Counter() for kind in RELATIONS}
    boundary = overlap_free = task_passes = task_total = 0
    for sample, old in zip(samples, legacy["details"], strict=True):
        prediction = by_id[sample["id"]]
        raw = prediction.get("raw_response", "")
        parsed, world, issues = None, None, []
        decoded_json_error = None
        if prediction.get("error"):
            issues.append({"code": "inference_error", "message": str(prediction["error"])})
        else:
            try:
                parsed = _strict_json(raw)
            except (TypeError, ValueError) as exc:
                issues.append({"code": "json_parse", "message": str(exc)})
            try:
                json.dumps(parsed, allow_nan=False)
            except ValueError:
                # Legal exponents such as 1e999 overflow Python floats. Keep raw losslessly.
                parsed = None
                decoded_json_error = "Numeric value exceeds finite float range; see raw_response"
            if old["json_parse"]:
                try:
                    world = WorldState.model_validate_json(raw)
                    validate_world(world)
                except ValidationError as exc:
                    issues.extend(
                        {"code": error["type"], "path": list(error["loc"]), "message": error["msg"]}
                        for error in exc.errors(include_input=False, include_url=False)
                    )
                except WorldValidationError as exc:
                    issues.extend(asdict(issue) for issue in exc.issues)
        codes = {issue["code"] for issue in issues}
        boundary_ok = world is not None and not codes.intersection(
            {"out_of_bounds", "duplicate_id"}
        )
        overlap_ok = world is not None and not codes.intersection(
            {"overlap", "blocked_entity", "duplicate_id"}
        )
        boundary += int(boundary_ok)
        overlap_free += int(overlap_ok)
        scores = relation_scores(world, sample)
        requirement_failures = []
        for kind, score in scores.items():
            relation_totals[kind].update({"passed": score["passed"], "total": score["total"]})
            if score["missing_slots"]:
                requirement_failures.append(
                    {
                        "code": "missing_required_relation",
                        "relation": kind,
                        "message": f"Missing {score['missing_slots']} required {kind} "
                        "relation slot(s)",
                        "missing_edges": score["missing_required_edges"],
                    }
                )
        task_ok = None
        if sample.get("required_objects"):
            from dataset_generation.spatial_tasks import check_task

            task_errors = check_task(world, sample) if world else ["No schema-valid world"]
            requirement_failures.extend(
                {"code": "task_constraint", "message": message} for message in task_errors
            )
            task_ok = old["expectations_pass"] and not task_errors
            task_total += 1
            task_passes += int(task_ok)
        annotated = [
            issue | {"category": CODE_CATEGORY.get(issue["code"], "other")} for issue in issues
        ]
        category_issues.update(issue["category"] for issue in annotated)
        category_samples.update({issue["category"] for issue in annotated})
        raw_codes.update(issue["code"] for issue in issues)
        prompt = sample.get("prompt") or sample["messages"][1]["content"]
        details.append(
            old
            | {
                "prompt": prompt,
                "raw_response": raw,
                "generated_json": parsed,
                "decoded_json_error": decoded_json_error,
                "validator_result": {
                    "evaluated": world is not None,
                    "passed": old["spatial_valid"],
                    "issues": annotated,
                },
                "failure_reasons": annotated,
                "relation_metrics": scores,
                "requirement_failures": requirement_failures,
                "task_constraints_pass": task_ok,
                "boundary_correct": boundary_ok,
                "overlap_free": overlap_ok,
                "footprints": [{"id": obj.id, "bounds": list(bounds(obj))} for obj in world.objects]
                if world
                else [],
            }
        )
    count = len(samples)
    extra = {
        "boundary_accuracy": boundary / count if count else None,
        "overlap_free_rate": overlap_free / count if count else None,
        "task_constraint_pass_rate": task_passes / task_total if task_total else None,
    }
    for kind, score in relation_totals.items():
        extra[f"{kind}_accuracy"] = score["passed"] / score["total"] if score["total"] else None
    return {
        "samples": count,
        "metrics": legacy["metrics"] | extra,
        "counts": legacy["counts"] | {"boundary_correct": boundary, "overlap_free": overlap_free},
        "relation_totals": relation_totals,
        "failure_issue_counts": {key: category_issues[key] for key in CATEGORIES},
        "failure_sample_counts": {key: category_samples[key] for key in CATEGORIES},
        "validator_code_counts": dict(sorted(raw_codes.items())),
        "definitions": legacy["definitions"]
        | {
            "relation_accuracy": "sum(valid unique predicted edges) / sum(max(predicted edges + "
            "missing explicitly requested edges, required minimum)). Invalid JSON contributes "
            "required slots but zero passes. Zero total is N/A. near/connected are symmetric; "
            "inside is directional. Duplicate edges add failures. Geometry only: an edge may "
            "pass while collision or boundary checks fail.",
            "legacy_relation_scope": "v0.4 prompts specify type minimums, not endpoint IDs; "
            "new benchmark uses required_relations to additionally penalize wrong endpoints.",
            "boundary_accuracy": "fraction of ALL samples with schema-valid worlds, unique IDs "
            "and every footprint within its declared map; map matching is a separate expectation",
            "overlap_free_rate": "fraction of ALL samples with schema-valid worlds, unique IDs, "
            "no overlap or blocked_entity; canonical layer and explicit containment rules apply",
            "failure_categories": "issue occurrences and unique sample counts are reported "
            "separately; blocked_entity maps to overlap; parse/schema/inference errors to other. "
            "Missing requested relations affect metrics, not canonical validator semantics.",
            "task_constraint_pass_rate": "among tasks with required_objects: full spatial and "
            "expectation pass plus exact IDs/types/sizes, required edges and requested boundary "
            "contact. N/A for legacy v0.4 samples without these annotations.",
        },
        "details": details,
    }


def write_analysis(root: Path, variants: dict[str, dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Spatial Failure Report",
        "",
        "Stored raw predictions; no repairs or retraining.",
        "",
    ]
    names = list(variants)
    lines += [
        "| Metric | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---:" for _ in names) + " |",
    ]
    for metric in next(iter(variants.values()))["metrics"]:
        values = [
            "N/A"
            if variants[name]["metrics"][metric] is None
            else f"{variants[name]['metrics'][metric]:.2%}"
            for name in names
        ]
        lines.append(f"| {metric} | " + " | ".join(values) + " |")
    for name, result in variants.items():
        write_json(
            root / f"{name}_spatial_metrics.json",
            {k: v for k, v in result.items() if k != "details"},
        )
        with (root / f"{name}_spatial_samples.jsonl").open("w", encoding="utf-8") as stream:
            for row in result["details"]:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        lines += [
            "",
            f"## {name}: {result['samples']} samples",
            "",
            "| Category | Issue occurrences | Affected samples |",
            "| --- | ---: | ---: |",
        ]
        for category in CATEGORIES:
            lines.append(
                f"| {category} | {result['failure_issue_counts'][category]} | "
                f"{result['failure_sample_counts'][category]} |"
            )
        lines += [
            "",
            "Original error codes: `" + json.dumps(result["validator_code_counts"]) + "`.",
            "",
            "Relation passes / slots: `" + json.dumps(result["relation_totals"]) + "`.",
            "",
            "### Exact failures",
            "",
        ]
        for row in result["details"]:
            if row["failure_reasons"]:
                lines.append(
                    f"- `{row['id']}`: "
                    + "; ".join(
                        f"{issue['code']}: {issue['message']}" for issue in row["failure_reasons"]
                    )
                )
            if row["requirement_failures"]:
                lines.append(
                    f"- `{row['id']}` prompt requirements: "
                    + "; ".join(item["message"] for item in row["requirement_failures"])
                )
        lines += [
            "",
            f"Full prompts, generated JSON and validator results: `{name}_spatial_samples.jsonl`.",
        ]
    lines += ["", "## Metric definitions", ""]
    lines += [
        f"- {name}: {value}" for name, value in next(iter(variants.values()))["definitions"].items()
    ]
    path = root / "spatial_failure_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit existing predictions without model loading")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation/v0_4_1"))
    args = parser.parse_args(argv)
    samples = read_jsonl(args.samples)
    results = {
        name: analyze(samples, read_jsonl(path))
        for name, path in (("base", args.base), ("lora", args.lora))
    }
    print(write_analysis(args.output_dir, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
