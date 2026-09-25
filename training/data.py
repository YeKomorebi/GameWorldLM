"""Chat-template token boundaries, assistant-only labels, and lossless context windows."""

import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]


def encode_chat(tokenizer, messages: list[dict]) -> tuple[list[int], int]:
    if [item["role"] for item in messages] != ["system", "user", "assistant"]:
        raise ValueError("Expected exactly system, user, assistant messages")
    prefix = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[: len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Chat template does not preserve an unambiguous assistant prefix")
    return full, len(prefix)


def assistant_windows(
    token_ids: list[int],
    assistant_start: int,
    max_length: int,
    overlap: int,
    policy: str,
) -> list[dict]:
    if not 0 < assistant_start < len(token_ids) or not 1 <= overlap < max_length:
        raise ValueError("Invalid assistant boundary or context overlap")
    if policy not in {"chunk_assistant", "error"}:
        raise ValueError("Unknown long-sequence policy")
    if len(token_ids) > max_length and policy == "error":
        raise ValueError(
            f"Sequence length {len(token_ids)} exceeds {max_length}; no truncation applied"
        )
    windows, start, covered = [], 0, assistant_start
    while start < len(token_ids):
        end = min(start + max_length, len(token_ids))
        first_label = max(assistant_start, covered, start + 1)
        if first_label < end:
            labels = [-100] * (first_label - start) + token_ids[first_label:end]
            windows.append(
                {
                    "input_ids": token_ids[start:end],
                    "attention_mask": [1] * (end - start),
                    "labels": labels,
                }
            )
            covered = end
        if end == len(token_ids):
            break
        start = end - overlap
    if (
        sum(sum(label != -100 for label in item["labels"]) for item in windows)
        != len(token_ids) - assistant_start
    ):
        raise ValueError("Assistant token coverage is incomplete")
    return windows


def tokenize_rows(tokenizer, rows: list[dict], settings) -> tuple[list[dict], dict]:
    examples, lengths = [], []
    for row in rows:
        tokens, boundary = encode_chat(tokenizer, row["messages"])
        lengths.append(len(tokens))
        examples.extend(
            assistant_windows(
                tokens,
                boundary,
                settings.max_seq_length,
                settings.overlap_tokens,
                settings.long_sequence_policy,
            )
        )
    return examples, {
        "samples": len(rows),
        "windows": len(examples),
        "long_samples": sum(length > settings.max_seq_length for length in lengths),
        "max_unwindowed_length": max(lengths, default=0),
        "supervised_tokens": sum(sum(label != -100 for label in row["labels"]) for row in examples),
        "truncated_tokens": 0,
    }


class AssistantCollator:
    def __init__(self, pad_token_id: int, multiple: int):
        self.pad_token_id, self.multiple = pad_token_id, multiple

    def __call__(self, rows):
        import torch

        length = max(len(row["input_ids"]) for row in rows)
        length = ((length + self.multiple - 1) // self.multiple) * self.multiple
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            padding = length - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            result["attention_mask"].append(row["attention_mask"] + [0] * padding)
            result["labels"].append(row["labels"] + [-100] * padding)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in result.items()}
