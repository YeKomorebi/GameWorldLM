"""Generate the frozen 30/40/30 evaluation inventory, never training labels."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from dataset.storage import schema_digest, write_json
from dataset_generation.prompt_generator import prompt_group
from dataset_generation.spatial_tasks import RELATIONS, check_task, make_task, world_fingerprint
from training.data import read_jsonl
from world.evaluation import GenerationExpectations, evaluate_response

DEFAULT_SEED = 20260925


def generate_benchmark(seed: int = DEFAULT_SEED) -> list[dict]:
    rows = []
    system = (
        (Path(__file__).resolve().parents[1] / "training/configs/system_prompt.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    for difficulty, count in (("easy", 30), ("medium", 40), ("hard", 30)):
        for _ in range(count):
            index = len(rows)
            row, witness = make_task(index, RELATIONS[index % 3], difficulty, seed, benchmark=True)
            _, report = evaluate_response(
                witness.model_dump_json(),
                GenerationExpectations.model_validate(row["expectations"]),
            )
            if not report.passed or check_task(witness, row):
                raise ValueError("Benchmark has no validated feasibility witness")
            rows.append(
                row
                | {
                    "prompt_group": prompt_group(row["prompt"]),
                    "schema_sha256": schema_digest(),
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": row["prompt"]},
                    ],
                    "feasibility": {
                        "validated": True,
                        "witness_sha256": world_fingerprint(witness),
                    },
                }
            )
    if len({row["prompt_group"] for row in rows}) != 100:
        raise ValueError("Duplicate benchmark prompt")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Freeze 100 held-out spatial benchmark prompts")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path("evaluation/prompts.jsonl"))
    parser.add_argument("--exclude-file", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    from .holdout import row_prompt

    rows = generate_benchmark(args.seed)
    excluded = {
        prompt_group(row_prompt(row)) for path in args.exclude_file for row in read_jsonl(path)
    }
    if excluded.intersection(row["prompt_group"] for row in rows):
        raise ValueError("Benchmark overlaps existing data")
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
        raise ValueError("Refusing to replace a different frozen benchmark; choose a new path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8", newline="\n")
    write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "version": "0.4.1",
            "seed": args.seed,
            "evaluation_only": True,
            "count": len(rows),
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "difficulty": dict(Counter(row["difficulty"] for row in rows)),
            "focus_relation": dict(Counter(row["focus_relation"] for row in rows)),
            "themes": dict(Counter(row["theme"] for row in rows)),
            "exclusion_files": [path.as_posix() for path in args.exclude_file],
            "feasibility": "Each task has a constructive witness passing the canonical validator; "
            "answers are not exported in the prompt inventory.",
            "difficulty_definition": {
                "easy": "one relation, 3 objects",
                "medium": "two shared-anchor/chain relations plus distractors, 5 objects",
                "hard": "mixed relation types, boundary contact, shared anchor or chain, 9 objects",
            },
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
