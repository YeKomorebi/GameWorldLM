"""Audit world distributions independently of model scores or training runs."""

import argparse
import json
from collections import Counter
from pathlib import Path

from dataset.storage import write_json
from dataset_generation.prompt_generator import prompt_group
from training.data import read_jsonl
from world.geometry import bounds
from world.schema import WorldState
from world.validator import WorldValidationError, validate_world

from .holdout import heldout_groups, row_prompt


def summarize(rows: list[dict]) -> dict:
    objects, relations, sizes, maps, sources = Counter(), Counter(), Counter(), Counter(), Counter()
    failures, samples_with_relation = Counter(), Counter()
    valid = boundary_contact = 0
    for row in rows:
        sources.update([row.get("source", "unknown")])
        try:
            world = WorldState.model_validate_json(row["messages"][-1]["content"])
            validate_world(world)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            codes = (
                {issue.code for issue in exc.issues}
                if isinstance(exc, WorldValidationError)
                else {"schema_or_record_error"}
            )
            failures.update(codes)
            continue
        valid += 1
        objects.update(obj.object_type for obj in world.objects)
        relations.update(rel.relation_type for rel in world.relations)
        samples_with_relation.update({rel.relation_type for rel in world.relations})
        sizes.update(f"{obj.attributes.size[0]}x{obj.attributes.size[1]}" for obj in world.objects)
        maps.update([f"{world.map.width}x{world.map.height}"])
        boundary_contact += int(
            any(
                bounds(obj)[0] == 0
                or bounds(obj)[1] == 0
                or bounds(obj)[2] == world.map.width
                or bounds(obj)[3] == world.map.height
                for obj in world.objects
            )
        )
    groups = [prompt_group(row_prompt(row)) for row in rows]
    return {
        "total_count": len(rows),
        "valid_count": valid,
        "failed_count": len(rows) - valid,
        "average_object_count": sum(objects.values()) / valid if valid else None,
        "object_distribution": dict(sorted(objects.items())),
        "relation_distribution": dict(sorted(relations.items())),
        "samples_with_relation": dict(sorted(samples_with_relation.items())),
        "footprint_distribution": dict(sorted(sizes.items())),
        "map_distribution": dict(sorted(maps.items())),
        "source_distribution": dict(sorted(sources.items())),
        "difficulty_distribution": dict(Counter(row.get("difficulty", "legacy") for row in rows)),
        "focus_relation_distribution": dict(
            Counter(row.get("focus_relation", "legacy") for row in rows)
        ),
        "theme_distribution": dict(Counter(row.get("theme", "legacy") for row in rows)),
        "failure_reason_distribution": dict(failures),
        "samples_touching_boundary": boundary_contact,
        "duplicate_normalized_prompts": len(groups) - len(set(groups)),
        "quality": "canonical_schema_and_spatial_checks; not_human_reviewed",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare original and curriculum data statistics")
    parser.add_argument("--prepared-dir", type=Path, default=Path("outputs/training/prepared"))
    parser.add_argument("--curriculum-dir", type=Path, default=Path("outputs/curriculum/v0_4_1"))
    parser.add_argument("--output", type=Path, default=Path("docs/v0_4_1_dataset_statistics.json"))
    args = parser.parse_args(argv)
    splits = {
        split: read_jsonl(args.prepared_dir / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    curriculum = read_jsonl(args.curriculum_dir / "train.jsonl")
    rejected = read_jsonl(args.curriculum_dir / "failed.jsonl")
    train = splits["train"] + curriculum
    protected = heldout_groups() | {
        prompt_group(row_prompt(row)) for row in splits["val"] + splits["test"]
    }
    overlap = {prompt_group(row_prompt(row)) for row in train}.intersection(protected)
    if overlap:
        raise ValueError("Candidate training data overlaps held-out evaluation prompts")
    report = {
        "v04_all": summarize(sum(splits.values(), [])),
        "v04_splits": {split: summarize(rows) for split, rows in splits.items()},
        "curriculum": summarize(curriculum)
        | {
            "total_count": len(curriculum) + len(rejected),
            "failed_count": len(rejected),
            "failure_reason_distribution": dict(Counter(row["failure_code"] for row in rejected)),
        },
        "candidate_train": summarize(train),
        "heldout_prompt_overlap": len(overlap),
        "retraining_started": False,
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                name: {key: value[key] for key in ("total_count", "valid_count", "failed_count")}
                for name, value in report.items()
                if name in ("v04_all", "curriculum", "candidate_train")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
