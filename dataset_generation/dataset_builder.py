"""Generate, validate, render, and checkpoint an instruction dataset incrementally."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from dataset.storage import schema_digest, utc_now, write_json
from llm.prompt_parser import build_messages
from llm.providers import ProviderConfig
from world.evaluation import evaluate_response

from .prompt_generator import (
    PROMPT_VERSION,
    PromptSpec,
    generate_prompts,
    inventory_summary,
    load_prompts,
    prompt_group,
    write_prompts,
)
from .qwen_generator import QwenGenerator
from .splits import DATASET_VERSION, export_records
from .statistics import Statistics, generation_duration
from .storage import Checkpoints, output_lock


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_result(case: PromptSpec, directory: Path, root: Path, identity: dict) -> dict:
    """Revalidate completed artifacts before admitting a sample to the training set."""
    paths = sorted((directory / "runs").glob("*/record.json"))
    if len(paths) > 1:
        raise ValueError(f"Ambiguous run history for {case.id}")
    record = read_json(paths[0]) if paths else {}
    run_dir = paths[0].parent if paths else None
    result = {
        "dataset_version": DATASET_VERSION,
        "id": case.id,
        "theme": case.theme,
        "prompt": case.prompt,
        "descriptors": case.descriptors,
        "expectations": case.expectations.model_dump(),
        "provider": identity["provider"],
        "model": identity["model"],
        "source": identity["source"],
        "schema_sha256": schema_digest(),
        "status": record.get("status", "interrupted"),
        "validation": record.get("validation", {"status": "not_evaluated", "passed": False}),
        "error": record.get("error"),
        "recorded_at": utc_now(),
        "run_id": record.get("run_id"),
        "artifacts": {},
        "attempts": len(record.get("attempts", [])),
        "reported_total_tokens": 0,
        "attempts_without_usage": 0,
        "actual_models": [],
        "http_status": None,
        "object_count": None,
        "generation_seconds": generation_duration(record),
        "quality": "automatically_validated_not_human_reviewed",
    }
    if result["status"] == "running":
        result["status"] = "interrupted"
    if result["status"] == "interrupted":
        result["error"] = (
            "Interrupted run; remote completion may have occurred. Not retried automatically."
        )
    actual_models = set()
    for attempt in record.get("attempts", []):
        provider = attempt.get("provider", {})
        usage = provider.get("usage") or {}
        total = usage.get("total_tokens")
        if isinstance(total, int) and total >= 0:
            result["reported_total_tokens"] += total
        else:
            result["attempts_without_usage"] += 1
        if provider.get("actual_model"):
            actual_models.add(provider["actual_model"])
        result["http_status"] = provider.get("http_status")
    result["actual_models"] = sorted(actual_models)
    if run_dir:
        for name, filename in (
            ("record", "record.json"),
            ("world", "world.json"),
            ("png", "map.png"),
            ("validation", "validation.json"),
        ):
            if (run_dir / filename).exists():
                result["artifacts"][name] = (run_dir / filename).relative_to(root).as_posix()
    if result["status"] == "success":
        try:
            if (
                record.get("source") != "llm"
                or identity["source"] != "llm"
                or record.get("schema_sha256") != schema_digest()
                or record.get("prompt") != case.prompt
                or record.get("case_id") != case.id
                or record.get("model") != identity["model"]
                or record.get("provider") != identity["provider"]
                or record.get("expectations") != case.expectations.model_dump()
            ):
                raise ValueError("Run provenance does not match this dataset")
            content = (run_dir / "world.json").read_bytes()
            if hashlib.sha256(content).hexdigest() != record.get("world_sha256"):
                raise ValueError("World file hash mismatch")
            world, validation = evaluate_response(content.decode("utf-8"), case.expectations)
            result["validation"] = validation.to_dict()
            if world is None or not validation.passed:
                raise ValueError("Stored world failed validation")
            with (run_dir / "map.png").open("rb") as image:
                if image.read(8) != b"\x89PNG\r\n\x1a\n":
                    raise ValueError("PNG signature is invalid")
            messages = read_json(run_dir / "attempts/001/request.json")["messages"]
            if messages != build_messages(case.prompt):
                raise ValueError("Initial request differs from the original instruction")
            result["messages"] = messages + [
                {"role": "assistant", "content": world.model_dump_json()}
            ]
            result["object_count"] = len(world.objects)
            result["object_counts"] = dict(Counter(obj.object_type for obj in world.objects))
            result["world_sha256"] = record["world_sha256"]
        except (ValueError, KeyError, OSError, TypeError) as exc:
            result.update(status="artifact_error", error=str(exc))
    errors = result["validation"].get("errors", [])
    result["failure_reasons"] = (
        []
        if result["status"] == "success"
        else sorted({str(error["code"]) for error in errors} or {result["status"]})
    )
    return result


class DatasetBuilder:
    def __init__(
        self,
        root: str | Path,
        generator: QwenGenerator,
        *,
        count=10_000,
        seed=42,
        cases: list[PromptSpec] | None = None,
    ):
        self.root = Path(root).resolve()
        self.generator = generator
        self.cases = generate_prompts(count, seed) if cases is None else cases
        if (
            not self.cases
            or len({case.id for case in self.cases}) != len(self.cases)
            or len({prompt_group(case.prompt) for case in self.cases}) != len(self.cases)
        ):
            raise ValueError(
                "Prompt cases must be nonempty, with unique IDs and normalized prompts"
            )
        self.seed = seed

    def run(self, *, limit: int | None = None, max_total_tokens: int | None = None) -> dict:
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        if max_total_tokens is not None and max_total_tokens < 1:
            raise ValueError("max_total_tokens must be positive")
        with output_lock(self.root):
            digest = write_prompts(self.root / "prompts.jsonl", self.cases)
            system_hash = hashlib.sha256(build_messages("scene")[0]["content"].encode()).hexdigest()
            contract = {
                "dataset_version": PROMPT_VERSION,
                "seed": self.seed,
                "count": len(self.cases),
                "prompts_sha256": digest,
                "schema_sha256": schema_digest(),
                "system_prompt_sha256": system_hash,
                "generator": self.generator.identity,
            }
            manifest_path = self.root / "manifest.json"
            if manifest_path.exists():
                if read_json(manifest_path)["contract"] != contract:
                    raise ValueError(
                        "Dataset configuration changed; "
                        "resume with original settings or a new directory"
                    )
            else:
                write_json(
                    manifest_path,
                    {
                        "created_at": utc_now(),
                        "contract": contract,
                        "inventory": inventory_summary(self.cases),
                    },
                )
            store = Checkpoints(self.root)
            try:
                report = self._run(store, limit, max_total_tokens)
                snapshot = export_records(
                    store.records(),
                    self.root,
                    source_root=self.root,
                    seed=self.seed,
                    planned=len(self.cases),
                )
                report["splits"] = snapshot["splits"]
                report["split_valid_count"] = snapshot["valid_count"]
                for name in (
                    "valid_count",
                    "failed_count",
                    "object_distribution",
                    "average_object_count",
                ):
                    report[name] = snapshot[name]
                report["export_rejected_count"] = snapshot["status_distribution"].get(
                    "export_rejected", 0
                )
                if report["export_rejected_count"]:
                    report["status"] = "export_failed"
                    report["stop_reason"] = (
                        "Training export rejected recorded successes; see failed.jsonl"
                    )
                write_json(self.root / "report.json", report)
                return report
            finally:
                store.close()

    def _run(self, store: Checkpoints, limit: int | None, max_tokens: int | None) -> dict:
        stats = Statistics(len(self.cases), self.root)
        states = store.states()
        by_id = {case.id: case for case in self.cases}
        if set(states) - by_id.keys():
            raise ValueError("Checkpoint contains IDs outside the prompt inventory")
        # Recover the narrow window between pipeline completion and checkpoint commit.
        for sample_id, state in states.items():
            if state == "running":
                case = by_id[sample_id]
                directory = self.root / "artifacts" / case.id
                store.finish(
                    sample_id, collect_result(case, directory, self.root, self.generator.identity)
                )
        for row in store.rebuild_jsonl():
            stats.add(row)
        report_path = self.root / "report.json"
        write_json(report_path, stats.report("running"))
        generated = consecutive_errors = 0
        reason = None
        status = "completed"
        try:
            for case in self.cases:
                if case.id in states:
                    continue
                if (self.root / "STOP").exists():
                    reason = "STOP file present"
                elif limit is not None and generated >= limit:
                    reason = "Invocation sample limit reached"
                elif max_tokens is not None and stats.tokens >= max_tokens:
                    reason = "Reported token threshold reached"
                if reason:
                    status = "paused"
                    break
                directory = self.root / "artifacts" / case.id
                store.start(case.id)
                if not directory.exists():
                    self.generator.generate(case, directory)
                row = collect_result(case, directory, self.root, self.generator.identity)
                store.finish(case.id, row)
                store.append_jsonl(row)
                stats.add(row)
                generated += 1
                write_json(report_path, stats.report("running"))
                print(
                    f"[{stats.succeeded + stats.failed}/{len(self.cases)}] {case.id}: "
                    f"{row['status']} (success={stats.succeeded}, failed={stats.failed})",
                    flush=True,
                )
                consecutive_errors = (
                    consecutive_errors + 1 if row["status"] == "provider_error" else 0
                )
                if row["http_status"] in {401, 403, 429}:
                    reason = f"Provider HTTP {row['http_status']}; remaining prompts pending"
                elif consecutive_errors >= 3:
                    reason = "Three consecutive provider errors; remaining prompts pending"
                elif row["status"] in {"render_error", "generation_error", "artifact_error"}:
                    reason = f"Pipeline infrastructure error: {row['status']}"
                if reason:
                    status = "paused"
                    break
        except (KeyboardInterrupt, SystemExit):
            status, reason = "interrupted", "Process interrupted"
            for sample_id, state in store.states().items():
                if state == "running":
                    case = by_id[sample_id]
                    directory = self.root / "artifacts" / sample_id
                    store.finish(
                        sample_id,
                        collect_result(case, directory, self.root, self.generator.identity),
                    )
            stats = Statistics(len(self.cases), self.root)
            for row in store.rebuild_jsonl():
                stats.add(row)
        except Exception:
            write_json(
                report_path, stats.report("error", "Builder failed; resume to recover checkpoints")
            )
            raise
        report = stats.report(status, reason)
        write_json(report_path, report)
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a resumable Qwen game world instruction dataset"
    )
    parser.add_argument(
        "--num-samples",
        "--count",
        dest="num_samples",
        type=int,
        help="Total prompts in this batch; defaults to 10000 or the input file size",
    )
    parser.add_argument("--prompts-file", type=Path, help="Existing PromptSpec JSONL or JSON array")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model")
    parser.add_argument("--attempts", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--tile-size", type=int, choices=range(16, 49), default=16)
    parser.add_argument("--requests-per-minute", type=float, default=20)
    parser.add_argument(
        "--limit", type=int, help="New samples in this invocation, excluding resume skips"
    )
    parser.add_argument(
        "--max-total-tokens", type=int, help="Reported token threshold checked between samples"
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Write prompts offline, without API calls"
    )
    args = parser.parse_args(argv)
    generator = None
    try:
        cases = (
            load_prompts(args.prompts_file, args.num_samples)
            if args.prompts_file is not None
            else generate_prompts(
                10_000 if args.num_samples is None else args.num_samples, args.seed
            )
        )
        if args.prepare_only:
            with output_lock(args.output_dir):
                digest = write_prompts(args.output_dir / "prompts.jsonl", cases)
                summary = inventory_summary(cases) | {"sha256": digest, "seed": args.seed}
                write_json(args.output_dir / "prompt_report.json", summary)
            print(json.dumps(summary, indent=2))
            return 0
        load_dotenv(args.env_file, override=False)
        generator = QwenGenerator(
            ProviderConfig.from_env("qwen", args.model),
            max_attempts=args.attempts,
            tile_size=args.tile_size,
            requests_per_minute=args.requests_per_minute,
        )
        report = DatasetBuilder(args.output_dir, generator, cases=cases, seed=args.seed).run(
            limit=args.limit,
            max_total_tokens=args.max_total_tokens,
        )
        print(json.dumps(report, indent=2))
        if report["status"] == "interrupted":
            return 130
        return (
            0
            if (
                report["status"] == "completed"
                and not report["failed"]
                and not report["export_rejected_count"]
            )
            else 1
        )
    except (ValueError, OSError) as exc:
        print(f"Dataset generation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if generator is not None:
            generator.close()


if __name__ == "__main__":
    raise SystemExit(main())
