"""Shared generation, object-frequency, and duration metrics for dataset reports."""

import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import get_args

from world.schema import ObjectType

from .prompt_generator import THEMES


def generation_duration(record: dict) -> float | None:
    try:
        started = datetime.fromisoformat(record["started_at"])
        finished = datetime.fromisoformat(record["finished_at"])
        seconds = (finished - started).total_seconds()
        return seconds if math.isfinite(seconds) and seconds > 0 else None
    except (KeyError, ValueError, TypeError):
        return None


class Statistics:
    def __init__(self, planned: int, source_root: Path | None = None):
        self.planned = planned
        self.source_root = source_root
        self.succeeded = self.failed = self.objects = self.attempts = self.tokens = 0
        self.unknown_usage = self.timed_samples = 0
        self.generation_seconds = 0.0
        self.reasons: Counter = Counter()
        self.statuses: Counter = Counter()
        self.object_distribution = Counter(dict.fromkeys(get_args(ObjectType), 0))
        self.themes = {theme: {"succeeded": 0, "failed": 0} for theme in THEMES}

    def add(self, row: dict) -> None:
        success = row["status"] == "success"
        self.succeeded += int(success)
        self.failed += int(not success)
        if success:
            counts = row.get("object_counts")
            if counts is None:
                world = json.loads(row["messages"][-1]["content"])
                counts = Counter(obj["object_type"] for obj in world["objects"])
            self.objects += sum(counts.values())
            self.object_distribution.update(counts)
        self.attempts += row["attempts"]
        self.tokens += row["reported_total_tokens"]
        self.unknown_usage += row["attempts_without_usage"]
        self.reasons.update(row["failure_reasons"])
        self.statuses.update([row["status"]])
        self.themes[row["theme"]]["succeeded" if success else "failed"] += 1
        seconds = row.get("generation_seconds")
        if seconds is None and self.source_root and row.get("artifacts", {}).get("record"):
            try:
                path = self.source_root / row["artifacts"]["record"]
                seconds = generation_duration(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                pass
        if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0:
            self.timed_samples += 1
            self.generation_seconds += seconds

    def report(self, status: str, reason: str | None = None) -> dict:
        processed = self.succeeded + self.failed
        pending = self.planned - processed
        average = self.generation_seconds / self.timed_samples if self.timed_samples else None
        remaining = average * pending if average is not None else None
        now = datetime.now(timezone.utc)
        eta = now + timedelta(seconds=remaining) if status == "running" and remaining else None
        return {
            "updated_at": now.isoformat(),
            "status": status,
            "stop_reason": reason,
            "planned": self.planned,
            "total_count": self.planned,
            "processed": processed,
            "pending": pending,
            "succeeded": self.succeeded,
            "valid_count": self.succeeded,
            "failed": self.failed,
            "failed_count": self.failed,
            "success_rate": self.succeeded / processed if processed else None,
            "average_object_count": self.objects / self.succeeded if self.succeeded else None,
            "object_distribution": dict(sorted(self.object_distribution.items())),
            "failure_reason_distribution": dict(sorted(self.reasons.items())),
            "status_distribution": dict(sorted(self.statuses.items())),
            "themes": self.themes,
            "sdk_attempts": self.attempts,
            "reported_total_tokens": self.tokens,
            "attempts_without_usage": self.unknown_usage,
            "timed_samples": self.timed_samples,
            "average_generation_seconds": average,
            "estimated_remaining_seconds": 0.0 if pending == 0 else remaining,
            "estimated_completion_at": eta.isoformat() if eta else None,
            "metric_definitions": {
                "total_count": "planned instructions, including pending and failed samples",
                "success_rate": "successful / processed; pending excluded",
                "average_object_count": "objects in successful worlds / successful samples",
                "object_distribution": "object tokens by type across valid worlds only",
                "failure_reason_distribution": (
                    "samples per distinct error code; multiple codes possible"
                ),
                "reported_total_tokens": "sum of reported usage; not an exact billing total",
                "sdk_attempts": "logical SDK calls; SDK transport retries may add HTTP requests",
                "estimated_completion_at": (
                    "UTC; serial average sample duration times pending; only while running; "
                    "assumes continuous service and adequate quota"
                ),
            },
        }


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .storage import read_snapshot

    parser = argparse.ArgumentParser(
        description="Read dataset statistics and a live completion estimate"
    )
    parser.add_argument("--source-dir", type=Path, default=Path("dataset"))
    args = parser.parse_args(argv)
    try:
        root = args.source_dir.resolve()
        rows = read_snapshot(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        previous = json.loads((root / "report.json").read_text(encoding="utf-8"))
        stats = Statistics(manifest["contract"]["count"], root)
        for row in rows:
            stats.add(row)
        print(json.dumps(stats.report(previous["status"], previous.get("stop_reason")), indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Dataset statistics failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
