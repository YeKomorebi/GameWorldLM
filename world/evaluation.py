"""Generation checks and optional prompt expectations, outside the world schema."""

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from world.geometry import bounds, shares_edge, tile_distance
from world.schema import MapSpec, ObjectType, WorldState
from world.validator import WorldValidationError, validate_world


class GenerationExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    object_counts: dict[ObjectType, Annotated[int, Field(ge=0, le=256)]] = Field(
        default_factory=dict
    )
    map: MapSpec | None = None
    minimum_relations: dict[
        Literal["near", "inside", "connected_to"], Annotated[int, Field(ge=0, le=512)]
    ] = Field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    status: str
    schema_valid: bool | None = None
    spatial_valid: bool | None = None
    expectations_valid: bool | None = None
    errors: list[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    repair_hints: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "valid"

    def to_dict(self) -> dict:
        return asdict(self) | {"passed": self.passed}

    def feedback(self) -> str:
        text = "\n".join(f"{error['code']}: {error['message']}" for error in self.errors)
        if self.context:
            text += "\nComputed footprint bounds [left,top,right,bottom], exclusive edges:\n"
            text += json.dumps(self.context, ensure_ascii=False)
        if self.repair_hints:
            text = "\n".join(self.repair_hints) + "\n" + text
        return text


def evaluate_response(
    raw: str, expectations: GenerationExpectations | None = None
) -> tuple[WorldState | None, ValidationReport]:
    try:
        world = WorldState.model_validate_json(raw)
    except ValidationError as exc:
        errors = [
            {"code": error["type"], "path": list(error["loc"]), "message": error["msg"]}
            for error in exc.errors(include_input=False, include_url=False)
        ]
        return None, ValidationReport("invalid_schema", schema_valid=False, errors=errors)
    try:
        validate_world(world)
    except WorldValidationError as exc:
        hints = []
        if expectations is not None:
            objects = {obj.id: obj for obj in world.objects}
            for rel in world.relations:
                if rel.source not in objects or rel.target not in objects:
                    continue
                source, target = objects[rel.source], objects[rel.target]
                invalid = (
                    rel.relation_type == "connected_to" and not shares_edge(source, target)
                ) or (rel.relation_type == "near" and tile_distance(source, target) > 3)
                if invalid and not expectations.minimum_relations.get(rel.relation_type):
                    hints.append(
                        f"Remove the invalid optional {rel.relation_type} relation "
                        f"from {rel.source} to {rel.target}. This test requires no "
                        f"{rel.relation_type} relations. Do not alter unrelated objects to keep it."
                    )
        return world, ValidationReport(
            "invalid_spatial",
            schema_valid=True,
            spatial_valid=False,
            errors=[asdict(issue) for issue in exc.issues],
            context={obj.id: list(bounds(obj)) for obj in world.objects},
            repair_hints=hints,
        )
    errors = []
    if expectations is not None:
        counts = Counter(obj.object_type for obj in world.objects)
        for kind, expected in expectations.object_counts.items():
            if counts[kind] != expected:
                errors.append(
                    {
                        "code": "object_count",
                        "message": f"{kind}: expected {expected}, got {counts[kind]}",
                    }
                )
        if expectations.map is not None and world.map != expectations.map:
            errors.append(
                {
                    "code": "map_mismatch",
                    "message": f"Expected map {expectations.map.model_dump()}",
                }
            )
        relations = Counter(rel.relation_type for rel in world.relations)
        for kind, minimum in expectations.minimum_relations.items():
            if relations[kind] < minimum:
                errors.append(
                    {
                        "code": "relation_count",
                        "message": f"{kind}: expected at least {minimum}, got {relations[kind]}",
                    }
                )
    return world, ValidationReport(
        "invalid_expectations" if errors else "valid",
        schema_valid=True,
        spatial_valid=True,
        expectations_valid=not errors if expectations is not None else None,
        errors=errors,
    )
