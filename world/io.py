"""JSON persistence with validation at both boundaries."""

from pathlib import Path

from world.schema import WorldState
from world.validator import validate_world


def load_world(path: str | Path) -> WorldState:
    return validate_world(WorldState.model_validate_json(Path(path).read_text(encoding="utf-8")))


def save_world(world: WorldState, path: str | Path) -> Path:
    validate_world(world)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(world.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return target
