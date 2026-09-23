"""One immutable directory per run, with atomic updates to its audit record."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from world.generator import GenerationAttempt
from world.schema import WorldState


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:12]


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def schema_digest() -> str:
    canonical = json.dumps(WorldState.model_json_schema(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RunWriter:
    def __init__(self, root: Path, metadata: dict):
        self.run_id = new_id()
        self.directory = root / "runs" / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.record = metadata | {
            "record_version": "1.0",
            "run_id": self.run_id,
            "started_at": utc_now(),
            "finished_at": None,
            "status": "running",
            "schema_version": "1.0",
            "schema_sha256": schema_digest(),
            "attempts": [],
            "artifacts": {"world": None, "png": None},
        }
        (self.directory / "prompt.txt").write_text(metadata["prompt"], encoding="utf-8")
        write_json(self.directory / "record.json", self.record)
        write_json(self.directory / "validation.json", {"status": "pending", "passed": False})

    def save_attempt(self, attempt: GenerationAttempt, provider_metadata: dict) -> None:
        directory = self.directory / "attempts" / f"{attempt.number:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "request.json", {"messages": attempt.messages})
        if attempt.raw_response is not None:
            (directory / "response.txt").write_text(attempt.raw_response, encoding="utf-8")
        write_json(directory / "validation.json", attempt.validation.to_dict())
        write_json(directory / "provider.json", provider_metadata)
        self.record["attempts"].append(
            {
                "number": attempt.number,
                "status": attempt.validation.status,
                "directory": f"attempts/{attempt.number:03d}",
                "provider": provider_metadata,
            }
        )
        write_json(self.directory / "record.json", self.record)

    def finish(self, status: str, validation: dict, *, error: str | None = None) -> None:
        world_path = self.directory / "world.json"
        png_path = self.directory / "map.png"
        self.record.update(
            {
                "status": status,
                "finished_at": utc_now(),
                "error": error,
                "validation": validation,
                "artifacts": {
                    "world": "world.json" if world_path.exists() else None,
                    "png": "map.png" if png_path.exists() else None,
                },
                "world_sha256": hashlib.sha256(world_path.read_bytes()).hexdigest()
                if world_path.exists()
                else None,
            }
        )
        write_json(self.directory / "validation.json", validation)
        write_json(self.directory / "record.json", self.record)
