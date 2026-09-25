"""Revalidate real worlds and freeze a reproducible chat dataset for local training."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from dataset.storage import schema_digest, utc_now, write_json
from dataset_generation.prompt_generator import prompt_group
from dataset_generation.splits import _artifact, _write_jsonl
from dataset_generation.storage import read_snapshot
from world.evaluation import GenerationExpectations, evaluate_response
from world.schema import WorldState

from .config import ExperimentConfig, load_config
from .data import read_jsonl


def validate_record(row: dict, root: Path) -> dict:
    if row.get("source") != "llm" or row.get("schema_sha256") != schema_digest():
        raise ValueError("Not a real LLM record with the current schema")
    path = _artifact(root, row, "record")
    record = json.loads(path.read_text(encoding="utf-8"))
    request = json.loads((path.parent / "attempts/001/request.json").read_text(encoding="utf-8"))
    messages = row["messages"]
    if (
        len(messages) != 3
        or [message["role"] for message in messages] != ["system", "user", "assistant"]
        or request["messages"] != messages[:2]
        or messages[1]["content"] != row["prompt"].strip()
        or record.get("case_id") != row["id"]
        or record.get("status") != "success"
        or record.get("source") != "llm"
        or record.get("prompt") != row["prompt"]
        or record.get("expectations") != row["expectations"]
        or record.get("schema_sha256") != schema_digest()
    ):
        raise ValueError("Source messages or audit record do not match the sample")
    content = _artifact(root, row, "world").read_bytes()
    if hashlib.sha256(content).hexdigest() != row["world_sha256"] or row[
        "world_sha256"
    ] != record.get("world_sha256"):
        raise ValueError("World artifact checksum differs")
    expected = GenerationExpectations.model_validate(row["expectations"])
    world, report = evaluate_response(content.decode("utf-8"), expected)
    if world is None or not report.passed:
        raise ValueError("World failed schema, spatial or expectation validation")
    if WorldState.model_validate_json(messages[-1]["content"]) != world:
        raise ValueError("Training answer differs from the world artifact")
    with _artifact(root, row, "png").open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Missing PNG artifact")
    return row | {"object_counts": dict(Counter(obj.object_type for obj in world.objects))}


def source_rows(root: Path):
    if (root / "checkpoint.sqlite3").is_file():
        for row in read_snapshot(root):
            yield row, root
    elif (root / "runs").is_dir():
        for path in sorted((root / "runs").glob("*/record.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") != "success" or record.get("source") != "llm":
                continue
            directory = path.parent
            messages = json.loads(
                (directory / "attempts/001/request.json").read_text(encoding="utf-8")
            )["messages"]
            world = (directory / "world.json").read_text(encoding="utf-8")
            row = record | {
                "id": record["case_id"],
                "messages": messages + [{"role": "assistant", "content": world}],
                "artifacts": {
                    name: (directory / filename).relative_to(root).as_posix()
                    for name, filename in (
                        ("world", "world.json"),
                        ("png", "map.png"),
                        ("record", "record.json"),
                    )
                },
            }
            yield row, root
    elif (root / "valid.jsonl").is_file():
        for row in read_jsonl(root / "valid.jsonl"):
            yield row, (root / row.get("artifact_root", ".")).resolve()
    else:
        raise ValueError(f"No supported recorded dataset in {root}")


def partition_counts(total: int, ratios: list[float]) -> list[int]:
    exact = [total * ratio for ratio in ratios]
    counts = [int(value) for value in exact]
    order = sorted(range(3), key=lambda index: -(exact[index] - counts[index]))
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return counts


def prepare(config: ExperimentConfig) -> dict:
    root = config.path(config.data.output_dir)
    system = config.path(config.data.system_prompt_file).read_text(encoding="utf-8").strip()
    if not system:
        raise ValueError("System prompt cannot be empty")
    accepted, skipped, seen = [], [], set()
    for source in config.data.sources:
        for original, artifact_root in source_rows(config.path(source)):
            try:
                if original.get("status") != "success":
                    raise ValueError("Not a successful generation")
                row = validate_record(original, artifact_root)
                group = prompt_group(row["prompt"])
                if group in seen:
                    raise ValueError("Duplicate normalized prompt")
                seen.add(group)
                world = WorldState.model_validate_json(row["messages"][-1]["content"])
                accepted.append(
                    {
                        "id": group,
                        "original_id": row["id"],
                        "run_id": row["run_id"],
                        "prompt_group": group,
                        "source": "llm",
                        "schema_sha256": schema_digest(),
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": row["prompt"].strip()},
                            {"role": "assistant", "content": world.model_dump_json()},
                        ],
                        "expectations": row["expectations"],
                        "object_counts": row["object_counts"],
                        "provenance": {
                            "source_dir": source,
                            "artifacts": row["artifacts"],
                            "world_sha256": row["world_sha256"],
                            "model": row["model"],
                        },
                    }
                )
            except (ValueError, KeyError, TypeError, OSError) as exc:
                skipped.append({"source": source, "id": original.get("id"), "reason": str(exc)})
    accepted.sort(
        key=lambda row: hashlib.sha256(f"{config.data.seed}:{row['id']}".encode()).hexdigest()
    )
    available = len(accepted)
    if config.data.max_samples is not None:
        if available < config.data.max_samples:
            raise ValueError(
                f"Requested {config.data.max_samples} valid samples; only {available} available"
            )
        accepted = accepted[: config.data.max_samples]
    counts = partition_counts(len(accepted), config.data.split_ratios)
    if min(counts) < 1:
        raise ValueError("Dataset must populate train, val, and test")
    signature = hashlib.sha256(
        json.dumps(
            {
                "rows": accepted,
                "seed": config.data.seed,
                "ratios": config.data.split_ratios,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    if root.exists():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("dataset_sha256") != signature:
            raise ValueError("Prepared dataset differs; choose a new data.output_dir")
        verify_prepared(root)
        return manifest
    root.mkdir(parents=True, exist_ok=False)
    files, offset = {}, 0
    for split, count in zip(("train", "val", "test"), counts, strict=True):
        files[f"{split}.jsonl"] = _write_jsonl(
            root / f"{split}.jsonl", accepted[offset : offset + count]
        )
        offset += count
    _write_jsonl(root / "rejected.jsonl", skipped)
    objects = Counter()
    for row in accepted:
        objects.update(row["object_counts"])
    manifest = {
        "created_at": utc_now(),
        "dataset_sha256": signature,
        "schema_sha256": schema_digest(),
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "seed": config.data.seed,
        "ratios": config.data.split_ratios,
        "available_unique_valid": available,
        "selected": len(accepted),
        "excluded": len(skipped),
        "splits": dict(zip(("train", "val", "test"), counts, strict=True)),
        "files_sha256": files,
        "object_distribution": dict(sorted(objects.items())),
        "quality": "automatic_schema_and_spatial_validation_not_human_reviewed",
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def verify_prepared(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_sha256"] != schema_digest():
        raise ValueError("Prepared dataset schema differs")
    for name, checksum in manifest["files_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Prepared dataset file changed: {name}")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare validated GameWorldLM chat training data")
    parser.add_argument("--config", default="training/configs/qwen7b_lora.json")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(load_config(args.config)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
