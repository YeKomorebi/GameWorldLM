"""CLI orchestration only; domain logic lives in world/, llm/, and engine/."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from examples.scenarios import load_scenarios
from llm.providers import LLMError, OpenAICompatibleBackend, ProviderConfig
from world.generator import WorldGenerationError, WorldGenerator
from world.io import load_world, save_world
from world.schema import WorldState
from world.validator import WorldValidationError


def _provider_options(parser):
    parser.add_argument("--provider", choices=["openai", "qwen", "fixture"], default=None)
    parser.add_argument("--model", help="Override the provider's configured model")
    parser.add_argument("--attempts", type=int, default=3, choices=range(1, 6))


def _backend(args):
    if args.provider == "fixture":
        from llm.fixture_backend import FixtureBackend

        return FixtureBackend(), "fixture", "hand-authored"
    config = ProviderConfig.from_env(args.provider, args.model)
    return OpenAICompatibleBackend(config), config.provider, config.model


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _generate(args) -> int:
    from engine.pygame_renderer import PygameRenderer

    prompt = args.prompt or Path(args.prompt_file).read_text(encoding="utf-8")
    backend, provider, model = _backend(args)
    try:
        result = WorldGenerator(backend, args.attempts).generate(prompt)
        destination = Path(args.output_dir)
        save_world(result.world, destination / "world.json")
        PygameRenderer(args.tile_size).render(result.world, destination / "map.png")
        _write_json(
            destination / "generation.json",
            {
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "attempts": result.attempts,
                "source": "fixture" if provider == "fixture" else "llm",
            },
        )
        print(
            f"Generated {result.world.scene} via {provider}/{model} in {result.attempts} attempt(s)"
        )
        print(f"JSON: {destination / 'world.json'}\nPNG: {destination / 'map.png'}")
        return 0
    finally:
        if isinstance(backend, OpenAICompatibleBackend):
            backend.close()


def _examples(args) -> int:
    from engine.pygame_renderer import PygameRenderer

    backend, provider, model = _backend(args)
    destination = Path(args.output_dir)
    generator = WorldGenerator(backend, args.attempts)
    renderer = PygameRenderer(args.tile_size)
    report = {
        "provider": provider,
        "model": model,
        "source": "fixture" if provider == "fixture" else "llm",
        "scenarios": [],
    }
    failures = 0
    try:
        for scenario in load_scenarios():
            record = {"name": scenario.name, "prompt": scenario.prompt}
            try:
                result = generator.generate(scenario.prompt)
                counts = Counter(obj.object_type for obj in result.world.objects)
                mismatches = {
                    kind: {"expected": count, "actual": counts[kind]}
                    for kind, count in scenario.expected_counts.items()
                    if counts[kind] != count
                }
                save_world(result.world, destination / f"{scenario.name}.json")
                renderer.render(result.world, destination / f"{scenario.name}.png")
                record.update(
                    {
                        "status": "count_mismatch" if mismatches else "ok",
                        "attempts": result.attempts,
                        "object_counts": dict(counts),
                        "expected_counts": scenario.expected_counts,
                        "mismatches": mismatches,
                    }
                )
                failures += bool(mismatches)
                print(f"{scenario.name}: {record['status']}")
            except (ValueError, LLMError, WorldGenerationError, OSError) as exc:
                record.update({"status": "error", "error": str(exc)})
                failures += 1
                print(f"{scenario.name}: error: {exc}", file=sys.stderr)
            report["scenarios"].append(record)
        _write_json(destination / "manifest.json", report)
        print(f"{5 - failures}/5 scenarios passed. Report: {destination / 'manifest.json'}")
        return 1 if failures else 0
    finally:
        if isinstance(backend, OpenAICompatibleBackend):
            backend.close()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="gameworldlm", description="Spatial tokens to 2D world maps"
    )
    root.add_argument("--env-file", default=".env", help="Optional dotenv file (existing env wins)")
    commands = root.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate and render a scene through an LLM")
    source = generate.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file")
    _provider_options(generate)
    generate.add_argument("--output-dir", default="outputs/generated")
    generate.add_argument("--tile-size", type=int, default=24)

    examples = commands.add_parser("examples", help="Generate the five evaluation scenarios")
    _provider_options(examples)
    examples.add_argument("--output-dir", default="outputs/examples")
    examples.add_argument("--tile-size", type=int, default=24)

    render = commands.add_parser("render", help="Validate and render an existing World JSON")
    render.add_argument("world")
    render.add_argument("--output", required=True)
    render.add_argument("--tile-size", type=int, default=24)

    validate = commands.add_parser("validate", help="Validate schema and spatial rules")
    validate.add_argument("world")
    schema = commands.add_parser("schema", help="Export the canonical JSON schema")
    schema.add_argument("--output", default="outputs/world.schema.json")
    assets = commands.add_parser("assets", help="Regenerate original placeholder PNG sprites")
    assets.add_argument("--output-dir", default="assets/placeholder")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    load_dotenv(args.env_file, override=False)
    try:
        if args.command == "generate":
            return _generate(args)
        if args.command == "examples":
            return _examples(args)
        if args.command == "render":
            from engine.pygame_renderer import PygameRenderer

            path = PygameRenderer(args.tile_size).render(load_world(args.world), args.output)
            print(f"PNG: {path}")
        elif args.command == "validate":
            world = load_world(args.world)
            print(
                f"Valid: {world.scene}, {len(world.objects)} objects, "
                f"{len(world.relations)} relations"
            )
        elif args.command == "schema":
            _write_json(Path(args.output), WorldState.model_json_schema())
            print(f"Schema: {args.output}")
        elif args.command == "assets":
            from engine.pygame_renderer import export_placeholder_assets

            paths = export_placeholder_assets(args.output_dir)
            print(f"Exported {len(paths)} sprites to {args.output_dir}")
        return 0
    except (
        ValidationError,
        WorldValidationError,
        WorldGenerationError,
        LLMError,
        ValueError,
        OSError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
