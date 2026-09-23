"""Revalidate recorded LLM runs and export leakage-aware messages/Alpaca JSONL."""

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from pathlib import Path

from dataset.storage import schema_digest, utc_now, write_json
from world.evaluation import GenerationExpectations, evaluate_response


def _prompt_group(prompt: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", prompt).split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def export_dataset(
    run_root: str | Path,
    destination: str | Path,
    *,
    validation_fraction: float = 0.2,
    seed: str = "gameworldlm-v0.2",
) -> dict:
    if not math.isfinite(validation_fraction) or not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    root = Path(run_root)
    if not (root / "runs").is_dir():
        raise ValueError(f"No runs directory in {root}")
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=False)
    records, samples, skipped = [], [], []
    duplicates = 0
    seen = set()
    for path in sorted((root / "runs").glob("*/record.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("Record is not a JSON object")
            records.append(record)
            if record.get("source") != "llm" or record.get("status") != "success":
                raise ValueError("Not a successful real-LLM run")
            if record.get("schema_sha256") != schema_digest():
                raise ValueError("World schema hash differs from this exporter")
            world_path = path.parent / "world.json"
            content = world_path.read_bytes()
            if hashlib.sha256(content).hexdigest() != record.get("world_sha256"):
                raise ValueError("World file hash mismatch")
            if not (path.parent / "map.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Missing or invalid PNG signature")
            expectations = (
                GenerationExpectations.model_validate(record["expectations"])
                if record.get("expectations") is not None
                else None
            )
            world, validation = evaluate_response(content.decode("utf-8"), expectations)
            if not validation.passed or world is None:
                raise ValueError("World no longer passes validation")
            request = json.loads(
                (path.parent / "attempts/001/request.json").read_text(encoding="utf-8")
            )
            messages = request["messages"]
            if (
                len(messages) != 2
                or [item["role"] for item in messages] != ["system", "user"]
                or not all(isinstance(item["content"], str) for item in messages)
                or messages[1]["content"] != record["prompt"].strip()
            ):
                raise ValueError("Original request does not match the recorded prompt")
            group = _prompt_group(record["prompt"])
            canonical = json.dumps(world.model_dump(), sort_keys=True, separators=(",", ":"))
            dedup_key = hashlib.sha256((group + canonical).encode("utf-8")).hexdigest()
            if dedup_key in seen:
                duplicates += 1
                continue
            seen.add(dedup_key)
            answer = world.model_dump_json()
            samples.append(
                {
                    "run_id": record["run_id"],
                    "group": group,
                    "messages": messages + [{"role": "assistant", "content": answer}],
                    "alpaca": {
                        "instruction": record["prompt"].strip(),
                        "input": "",
                        "output": answer,
                        "system": messages[0]["content"],
                    },
                }
            )
        except (ValueError, KeyError, TypeError, UnicodeError, OSError) as exc:
            skipped.append({"run_id": path.parent.name, "reason": str(exc)})

    # Split prompt groups, not individual completions, so repeated prompts never leak.
    groups = sorted(
        {item["group"] for item in samples},
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )
    validation_count = (
        min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
        if validation_fraction > 0 and len(groups) > 1
        else 0
    )
    validation_groups = set(groups[:validation_count])
    membership = []
    counts = {}
    for split in ("train", "validation"):
        selected = [
            item
            for item in samples
            if (item["group"] in validation_groups) == (split == "validation")
        ]
        _jsonl(output / f"{split}.jsonl", [{"messages": item["messages"]} for item in selected])
        _jsonl(output / f"alpaca_{split}.jsonl", [item["alpaca"] for item in selected])
        membership.extend(
            {"run_id": item["run_id"], "prompt_group": item["group"], "split": split}
            for item in selected
        )
        counts[split] = len(selected)
    _jsonl(output / "records.jsonl", records)
    _jsonl(output / "membership.jsonl", membership)
    dataset_info = {}
    for split in ("train", "validation"):
        dataset_info[f"gameworldlm_{split}"] = {
            "file_name": f"{split}.jsonl",
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
        dataset_info[f"gameworldlm_alpaca_{split}"] = {
            "file_name": f"alpaca_{split}.jsonl",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        }
    write_json(output / "dataset_info.json", dataset_info)
    manifest = {
        "dataset_version": "1.0",
        "created_at": utc_now(),
        "schema_sha256": schema_digest(),
        "source": "llm",
        "records": len(records),
        "accepted": len(samples),
        "duplicates_removed": duplicates,
        "skipped": skipped,
        "splits": counts,
        "prompt_groups": len(groups),
        "validation_fraction": validation_fraction,
        "seed": seed,
        "split_policy": "NFKC/whitespace/case-normalized prompt groups",
        "quality": "automatically_validated_not_human_reviewed",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export validated GameWorldLM runs for SFT/LoRA")
    parser.add_argument("--runs", default="outputs/real")
    parser.add_argument(
        "--output", required=True, help="New destination directory; never overwritten"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", default="gameworldlm-v0.2")
    args = parser.parse_args(argv)
    try:
        report = export_dataset(
            args.runs, args.output, validation_fraction=args.validation_fraction, seed=args.seed
        )
    except (ValueError, OSError) as exc:
        print(f"Dataset export failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Exported {report['accepted']} samples: {report['splits']}; "
        f"skipped {len(report['skipped'])}"
    )
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
