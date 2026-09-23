"""Revalidated, reproducible 80/10/10 training snapshots from committed samples."""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

from dataset.storage import schema_digest, utc_now, write_json
from llm.prompt_parser import build_messages
from world.evaluation import GenerationExpectations, evaluate_response

from .prompt_generator import prompt_group
from .statistics import Statistics, generation_duration
from .storage import output_lock, read_snapshot

DATASET_VERSION = "0.3.1"
SPLIT_WEIGHTS = {"train": 80, "val": 10, "test": 10}


def split_sizes(total: int) -> dict[str, int]:
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("total must be a nonnegative integer")
    sizes = {name: total * weight // 100 for name, weight in SPLIT_WEIGHTS.items()}
    remainder_order = sorted(SPLIT_WEIGHTS, key=lambda name: -(total * SPLIT_WEIGHTS[name] % 100))
    for name in remainder_order[: total - sum(sizes.values())]:
        sizes[name] += 1
    return sizes


def _artifact(root: Path, row: dict, name: str) -> Path:
    relative = Path(row["artifacts"][name])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path leaves the source dataset")
    return path


def _validate_success(row: dict, source: Path) -> dict:
    if row.get("source") != "llm" or row.get("schema_sha256") != schema_digest():
        raise ValueError("Sample is not a real LLM result with the current schema")
    messages = row["messages"]
    if (
        len(messages) != 3
        or [item["role"] for item in messages] != ["system", "user", "assistant"]
        or messages[:2] != build_messages(row["prompt"])
    ):
        raise ValueError("Training messages do not match the original instruction")
    expectations = GenerationExpectations.model_validate(row["expectations"])
    world, validation = evaluate_response(messages[-1]["content"], expectations)
    if world is None or not validation.passed:
        raise ValueError("Assistant world failed schema, spatial, or expectation validation")
    content = _artifact(source, row, "world").read_bytes()
    if hashlib.sha256(content).hexdigest() != row.get("world_sha256"):
        raise ValueError("World artifact checksum mismatch")
    recorded_world, recorded_validation = evaluate_response(content.decode("utf-8"), expectations)
    if not recorded_validation.passed or recorded_world != world:
        raise ValueError("Assistant answer differs from the validated world artifact")
    with _artifact(source, row, "png").open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Missing or invalid PNG artifact")
    record = json.loads(_artifact(source, row, "record").read_text(encoding="utf-8"))
    if (
        record.get("status") != "success"
        or record.get("source") != "llm"
        or record.get("case_id") != row["id"]
        or record.get("prompt") != row["prompt"]
        or record.get("model") != row["model"]
        or record.get("schema_sha256") != row["schema_sha256"]
        or record.get("expectations") != row["expectations"]
    ):
        raise ValueError("Audit record differs from the training sample")
    return row | {
        "validation": validation.to_dict(),
        "object_count": len(world.objects),
        "object_counts": dict(Counter(obj.object_type for obj in world.objects)),
        "generation_seconds": generation_duration(record),
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as stream:
        for row in rows:
            content = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            stream.write(content)
            digest.update(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return digest.hexdigest()


def export_records(
    records: Iterable[dict],
    destination: Path,
    *,
    source_root: Path,
    seed: int = 42,
    planned: int | None = None,
) -> dict:
    """Caller owns the destination lock; the source artifacts remain read-only."""
    rows, groups, ids = [], set(), set()
    try:
        artifact_root = Path(os.path.relpath(source_root, destination)).as_posix()
    except ValueError:
        artifact_root = source_root.resolve().as_posix()
    for original in records:
        row = deepcopy(original)
        if row["id"] in ids:
            raise ValueError(f"Duplicate sample ID: {row['id']}")
        ids.add(row["id"])
        if row["status"] == "success":
            try:
                row = _validate_success(row, source_root)
                group = prompt_group(row["prompt"])
                if group in groups:
                    raise ValueError("Duplicate normalized prompt")
                groups.add(group)
                row["prompt_group"] = group
            except (ValueError, KeyError, TypeError, OSError) as exc:
                row.update(
                    status="export_rejected",
                    error=str(exc),
                    failure_reasons=["export_rejected"],
                    object_count=None,
                )
        if row["status"] != "success":
            row.pop("messages", None)
            row.pop("object_counts", None)
        row["artifact_root"] = artifact_root
        rows.append(row)
    stats = Statistics(len(rows) if planned is None else planned, source_root)
    for row in rows:
        stats.add(row)
    accepted = [row for row in rows if row["status"] == "success"]
    ordered = sorted(
        accepted,
        key=lambda row: hashlib.sha256(f"{seed}:{row['prompt_group']}".encode()).hexdigest(),
    )
    counts = split_sizes(len(accepted))
    membership, offset = {}, 0
    for name, count in counts.items():
        membership.update((row["id"], name) for row in ordered[offset : offset + count])
        offset += count
    files = {}
    for name in SPLIT_WEIGHTS:
        files[f"{name}.jsonl"] = _write_jsonl(
            destination / f"{name}.jsonl",
            (row | {"split": name} for row in accepted if membership[row["id"]] == name),
        )
    files["valid.jsonl"] = _write_jsonl(destination / "valid.jsonl", accepted)
    files["failed.jsonl"] = _write_jsonl(
        destination / "failed.jsonl", (row for row in rows if row["status"] != "success")
    )
    files["split_membership.jsonl"] = _write_jsonl(
        destination / "split_membership.jsonl",
        (
            {"id": row["id"], "prompt_group": row["prompt_group"], "split": membership[row["id"]]}
            for row in accepted
        ),
    )
    write_json(
        destination / "dataset_info.json",
        {
            f"gameworldlm_{name}": {
                "file_name": f"{name}.jsonl",
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system",
                },
            }
            for name in SPLIT_WEIGHTS
        },
    )
    report = stats.report("snapshot") | {"splits": counts}
    write_json(destination / "statistics.json", report)
    write_json(
        destination / "split_manifest.json",
        {
            "dataset_version": DATASET_VERSION,
            "created_at": utc_now(),
            "schema_sha256": schema_digest(),
            "seed": seed,
            "splits": counts,
            "ratios": {name: weight / 100 for name, weight in SPLIT_WEIGHTS.items()},
            "rounding": "largest remainder; ties prefer train, then val, then test",
            "policy": "seeded SHA-256 order of unique normalized prompts",
            "artifact_root": artifact_root,
            "files_sha256": files,
            "total_count": report["total_count"],
            "processed": report["processed"],
            "valid_count": report["valid_count"],
            "failed_count": report["failed_count"],
        },
    )
    return report


def export_snapshot(
    source: str | Path,
    destination: str | Path,
    *,
    num_samples: int | None = None,
    seed: int = 42,
) -> dict:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError("Snapshot destination must be a new directory, separate from the source")
    rows = read_snapshot(source, num_samples)
    if not rows:
        raise ValueError("No finished samples to export")
    destination.mkdir(parents=True, exist_ok=False)
    with output_lock(destination):
        return export_records(rows, destination, source_root=source, seed=seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export an offline 80/10/10 training snapshot")
    parser.add_argument("--source-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-samples", type=int, help="First N finished records, including failures"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        report = export_snapshot(
            args.source_dir,
            args.output_dir,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    except (ValueError, OSError) as exc:
        print(f"Dataset split failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["valid_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
