"""Live API evaluation entry point. No fixture fallback and no embedded credentials."""

import argparse
import json
import sys
from importlib.resources import files
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from dataset.export import export_dataset
from dataset.storage import new_id, utc_now, write_json
from gameworldlm.pipeline import GenerationPipeline
from llm.providers import OpenAICompatibleBackend, ProviderConfig
from world.evaluation import GenerationExpectations
from world.schema import Identifier


class PromptCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: Identifier
    prompt: Annotated[str, Field(min_length=1, max_length=8000)]
    expectations: GenerationExpectations | None = None


def load_cases(path: str | Path | None = None) -> list[PromptCase]:
    text = (
        Path(path).read_text(encoding="utf-8")
        if path is not None
        else files("examples").joinpath("real_prompts.json").read_text(encoding="utf-8")
    )
    data = json.loads(text)
    if not isinstance(data, list) or not data:
        raise ValueError("Prompt file must contain a nonempty JSON array")
    cases = [PromptCase.model_validate(item) for item in data]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Prompt case IDs must be unique")
    if any(not case.prompt.strip() for case in cases):
        raise ValueError("Prompts cannot be blank")
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run real LLM generation and save audited datasets"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompt", help="Generate one custom scene instead of the ten-case suite")
    source.add_argument("--prompts-file", help="Custom JSON prompt suite")
    parser.add_argument("--provider", choices=["qwen", "openai"], default="qwen")
    parser.add_argument("--model")
    parser.add_argument(
        "--case-id", action="append", help="Select named suite cases; repeat to select several"
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default="outputs/real")
    parser.add_argument(
        "--limit", type=int, help="Run only the first N cases, e.g. 1 for a smoke test"
    )
    parser.add_argument("--attempts", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--tile-size", type=int, default=24)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv(args.env_file, override=False)
    backend = None
    try:
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        cases = (
            [PromptCase(id="custom", prompt=args.prompt)]
            if args.prompt is not None
            else load_cases(args.prompts_file)
        )
        if args.case_id:
            if args.prompt is not None:
                raise ValueError("--case-id cannot be combined with --prompt")
            unknown = set(args.case_id) - {case.id for case in cases}
            if unknown:
                raise ValueError(f"Unknown case IDs: {', '.join(sorted(unknown))}")
            cases = [case for case in cases if case.id in args.case_id]
        if args.limit:
            cases = cases[: args.limit]
        config = ProviderConfig.from_env(args.provider, args.model)
        backend = OpenAICompatibleBackend(config)
        root = Path(args.output_dir)
        pipeline = GenerationPipeline(
            backend,
            provider=config.provider,
            model=config.model,
            output_dir=root,
            max_attempts=args.attempts,
            tile_size=args.tile_size,
            parameters={
                "max_tokens": config.max_tokens,
                "timeout_seconds": config.timeout,
                "enable_thinking": config.enable_thinking if config.provider == "qwen" else None,
            },
        )
        batch_id = new_id()
        manifest_path = root / "batches" / f"{batch_id}.json"
        manifest = {
            "batch_id": batch_id,
            "started_at": utc_now(),
            "status": "running",
            "provider": config.provider,
            "model": config.model,
            "planned": len(cases),
            "runs": [],
        }
        write_json(manifest_path, manifest)
        for case in cases:
            print(f"Generating {case.id} via {config.provider}/{config.model}...", flush=True)
            result = pipeline.run(case.prompt, case_id=case.id, expectations=case.expectations)
            manifest["runs"].append(
                {"case_id": case.id, "run_id": result.run_id, "status": result.status}
            )
            write_json(manifest_path, manifest)
            print(f"{case.id}: {result.status}; {result.directory}", flush=True)
            if result.error:
                print(result.error, file=sys.stderr, flush=True)
            metadata = backend.last_metadata
            if metadata.get("http_status") in {401, 403, 429}:
                manifest["stopped_reason"] = (
                    f"Provider HTTP {metadata['http_status']}; remaining cases not called"
                )
                break
        success = sum(item["status"] == "success" for item in manifest["runs"])
        manifest.update(
            {
                "status": "success" if success == len(cases) else "failed",
                "finished_at": utc_now(),
                "succeeded": success,
                "not_run": len(cases) - len(manifest["runs"]),
            }
        )
        write_json(manifest_path, manifest)
        if not args.no_export:
            destination = root / "datasets" / batch_id
            dataset = export_dataset(root, destination)
            manifest["dataset"] = {"path": f"datasets/{batch_id}", "accepted": dataset["accepted"]}
            write_json(manifest_path, manifest)
            print(f"Dataset: {destination} ({dataset['accepted']} accepted samples)")
        print(f"{success}/{len(cases)} passed. Batch report: {manifest_path}")
        return 0 if success == len(cases) else 1
    except (ValueError, OSError) as exc:
        print(f"Real generation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
