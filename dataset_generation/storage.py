"""Single-writer checkpoints; JSONL files are rebuildable views of committed results."""

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def output_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".generation.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another dataset builder owns this output directory") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Checkpoints:
    def __init__(self, root: Path):
        self.root = root
        self.connection = sqlite3.connect(root / "checkpoint.sqlite3")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS samples "
            "(id TEXT PRIMARY KEY, state TEXT NOT NULL, result TEXT)"
        )
        self.connection.commit()

    def states(self) -> dict[str, str]:
        return dict(self.connection.execute("SELECT id, state FROM samples"))

    def start(self, sample_id: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO samples (id, state) VALUES (?, 'running')", (sample_id,)
            )

    def finish(self, sample_id: str, result: dict) -> None:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE samples SET state='finished', result=? WHERE id=? AND state='running'",
                (json.dumps(result, ensure_ascii=False), sample_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Sample has no running checkpoint: {sample_id}")

    def records(self) -> Iterator[dict]:
        for (result,) in self.connection.execute(
            "SELECT result FROM samples WHERE state='finished' ORDER BY rowid"
        ):
            yield json.loads(result)

    def rebuild_jsonl(self) -> Iterator[dict]:
        """Recover even if the previous process died during a JSONL append."""
        train = self.root / "train.jsonl.tmp"
        failed = self.root / "failed.jsonl.tmp"
        with train.open("w", encoding="utf-8", newline="\n") as good:
            with failed.open("w", encoding="utf-8", newline="\n") as bad:
                for record in self.records():
                    target = good if record["status"] == "success" else bad
                    target.write(json.dumps(record, ensure_ascii=False) + "\n")
                    yield record
                for stream in (good, bad):
                    stream.flush()
                    os.fsync(stream.fileno())
        train.replace(self.root / "train.jsonl")
        failed.replace(self.root / "failed.jsonl")

    def append_jsonl(self, record: dict) -> None:
        name = "train.jsonl" if record["status"] == "success" else "failed.jsonl"
        with (self.root / name).open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def close(self) -> None:
        self.connection.close()
