"""Construct, validate and render targeted synthetic spatial training candidates."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from dataset.storage import schema_digest, write_json
from engine.pygame_renderer import PygameRenderer
from evaluation.dataset_statistics import summarize
from evaluation.holdout import BENCHMARK_PATH, heldout_groups, reject_holdout
from training.data import read_jsonl
from world.evaluation import GenerationExpectations, evaluate_response
from world.schema import WorldState

from .prompt_generator import prompt_group
from .spatial_tasks import DIFFICULTIES, RELATIONS, check_task, make_task, world_fingerprint


def write_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    root: Path,
    *,
    num_samples: int = 270,
    seed: int = 42,
    relation: str | None = None,
    difficulty: str | None = None,
    prepared_dir: Path | None = None,
    benchmark: Path = BENCHMARK_PATH,
    tile_size: int = 16,
) -> dict:
    if (
        isinstance(num_samples, bool)
        or not isinstance(num_samples, int)
        or not 1 <= num_samples <= 500
    ):
        raise ValueError("Curriculum batch size must be between 1 and 500")
    if relation is not None and relation not in RELATIONS:
        raise ValueError("Unsupported relation")
    if difficulty is not None and difficulty not in DIFFICULTIES:
        raise ValueError("Unsupported difficulty")
    benchmark_rows = read_jsonl(benchmark)
    # The packaged registry is always protected, even when a custom benchmark is supplied.
    protected = heldout_groups([BENCHMARK_PATH, benchmark])
    known_worlds = {row["feasibility"]["witness_sha256"] for row in benchmark_rows}
    legacy_train = []
    if prepared_dir is not None:
        from training.prepare_dataset import verify_prepared

        verify_prepared(prepared_dir)
        protected |= heldout_groups([prepared_dir / "val.jsonl", prepared_dir / "test.jsonl"])
        legacy_train = read_jsonl(prepared_dir / "train.jsonl")
        for split in ("train", "val", "test"):
            for row in read_jsonl(prepared_dir / f"{split}.jsonl"):
                world = WorldState.model_validate_json(row["messages"][-1]["content"])
                known_worlds.add(world_fingerprint(world))
        for row in legacy_train:
            reject_holdout(row, protected)
    seen = protected | {prompt_group(row["messages"][1]["content"]) for row in legacy_train}
    system = (
        (Path(__file__).resolve().parents[1] / "training/configs/system_prompt.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    renderer = PygameRenderer(tile_size)
    root.mkdir(parents=True, exist_ok=False)
    plan = [
        (kind, level)
        for kind in ((relation,) if relation else RELATIONS)
        for level in ((difficulty,) if difficulty else DIFFICULTIES)
    ]
    accepted, failed = [], []
    for index in range(num_samples):
        kind, level = plan[index % len(plan)]
        row = {"id": f"pending_{index}", "focus_relation": kind, "difficulty": level}
        try:
            row, world = make_task(index, kind, level, seed)
            reject_holdout(row, protected)
            group, fingerprint = prompt_group(row["prompt"]), world_fingerprint(world)
            if group in seen or fingerprint in known_worlds:
                raise ValueError("Duplicate or held-out prompt/world")
            _, validation = evaluate_response(
                world.model_dump_json(), GenerationExpectations.model_validate(row["expectations"])
            )
            task_errors = check_task(world, row)
            if not validation.passed or task_errors:
                raise ValueError(f"Quality filter failed: {validation.errors}; {task_errors}")
            directory = root / "artifacts" / row["id"]
            directory.mkdir(parents=True)
            write_json(directory / "world.json", world.model_dump())
            write_json(
                directory / "validation.json", validation.to_dict() | {"task_errors": task_errors}
            )
            renderer.render(world, directory / "map.png")
            with (directory / "map.png").open("rb") as image:
                if image.read(8) != b"\x89PNG\r\n\x1a\n":
                    raise ValueError("Invalid render artifact")
            result = row | {
                "source": "synthetic_constructive",
                "model": None,
                "status": "success",
                "generator_version": "0.4.1",
                "schema_sha256": schema_digest(),
                "prompt_group": group,
                "world_fingerprint": fingerprint,
                "validation": validation.to_dict(),
                "task_errors": task_errors,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": world.model_dump_json()},
                ],
                "quality": "constructive_labels_automatically_validated_not_human_reviewed",
                "artifacts": {
                    name: (directory / filename).relative_to(root).as_posix()
                    for name, filename in (
                        ("world", "world.json"),
                        ("png", "map.png"),
                        ("validation", "validation.json"),
                    )
                },
                "artifact_sha256": {
                    filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                    for filename in ("world.json", "map.png", "validation.json")
                },
            }
            write_json(directory / "record.json", result)
            accepted.append(result)
            seen.add(group)
            known_worlds.add(fingerprint)
        except (ValueError, OSError) as exc:
            failed.append(
                row
                | {
                    "status": "failed",
                    "error": str(exc),
                    "failure_code": "duplicate_candidate"
                    if str(exc).startswith("Duplicate")
                    else "generation_or_artifact_error",
                    "source": "synthetic_constructive",
                }
            )
    hashes = {
        "train.jsonl": write_jsonl(root / "train.jsonl", accepted),
        "failed.jsonl": write_jsonl(root / "failed.jsonl", failed),
    }
    if legacy_train:
        # Artifact roots are explicit because old and new rows have different provenance.
        project = Path(__file__).resolve().parents[1]
        candidates = [
            row
            | {
                "artifact_root": Path(
                    os.path.relpath(project / row["provenance"]["source_dir"], root)
                ).as_posix()
            }
            for row in legacy_train
        ]
        candidates += [row | {"artifact_root": "."} for row in accepted]
        hashes["candidate_train.jsonl"] = write_jsonl(root / "candidate_train.jsonl", candidates)
    report = summarize(accepted) | {
        "version": "0.4.1",
        "seed": seed,
        "requested_count": num_samples,
        "total_count": num_samples,
        "failed_count": len(failed),
        "generation_failures": failed,
        "success_rate": len(accepted) / num_samples,
        "failure_reason_distribution": dict(Counter(row["failure_code"] for row in failed)),
        "candidate_train_count": len(legacy_train) + len(accepted),
        "schema_sha256": schema_digest(),
        "files_sha256": hashes,
        "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
        "heldout_prompt_overlap": 0,
        "protected_prompt_count": len(protected),
        "training_started": False,
        "model": None,
        "source": "synthetic_constructive",
        "note": "Curriculum rows are training candidates, never used to tune benchmark answers. "
        "v0.4 frozen val/test remain held out. No API calls or model training.",
    }
    write_json(root / "statistics.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a small validated spatial curriculum; no training"
    )
    parser.add_argument("--num-samples", type=int, default=270)
    parser.add_argument("--relation", choices=RELATIONS)
    parser.add_argument("--difficulty", choices=DIFFICULTIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/curriculum/v0_4_1"))
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--tile-size", type=int, default=16)
    args = parser.parse_args(argv)
    report = build(
        args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        relation=args.relation,
        difficulty=args.difficulty,
        prepared_dir=args.prepared_dir,
        benchmark=args.benchmark,
        tile_size=args.tile_size,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "requested_count",
                    "valid_count",
                    "failed_count",
                    "candidate_train_count",
                )
            },
            indent=2,
        )
    )
    return int(bool(report["failed_count"]))


if __name__ == "__main__":
    raise SystemExit(main())
