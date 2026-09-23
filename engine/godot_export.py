"""Reserved adapter boundary. Godot integration is intentionally outside MVP."""

from pathlib import Path
from typing import Protocol

from world.schema import WorldState


class WorldExporter(Protocol):
    def export(self, world: WorldState, destination: Path) -> Path: ...


class GodotExporter:
    def export(self, world: WorldState, destination: Path) -> Path:
        raise NotImplementedError(
            "Godot export is planned for phase 2. Use the pygame renderer for this MVP."
        )
