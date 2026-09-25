"""Reject frozen evaluation instructions at training-data admission boundaries."""

from pathlib import Path

from dataset_generation.prompt_generator import prompt_group
from training.data import read_jsonl

BENCHMARK_PATH = Path(__file__).with_name("prompts.jsonl")


def row_prompt(row: dict) -> str:
    if "prompt" in row:
        return row["prompt"]
    return next(message["content"] for message in row["messages"] if message["role"] == "user")


def heldout_groups(paths: list[Path] | None = None) -> set[str]:
    return {
        prompt_group(row_prompt(row))
        for path in (paths or [BENCHMARK_PATH])
        for row in read_jsonl(path)
    }


def reject_holdout(row: dict, groups: set[str]) -> None:
    if row.get("evaluation_only") or prompt_group(row_prompt(row)) in groups:
        raise ValueError("Frozen evaluation prompt cannot be admitted to training")
