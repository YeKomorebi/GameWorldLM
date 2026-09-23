"""Canonical, versioned data contract shared by every pipeline stage."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

ObjectType = Literal[
    "house",
    "tree",
    "river",
    "road",
    "bridge",
    "monster",
    "castle",
    "ruin",
    "rock",
    "dune",
    "tower",
    "wall",
    "library",
    "portal",
    "npc",
    "crystal",
    "courtyard",
    "vehicle",
]
Biome = Literal["forest", "snow", "desert", "city", "arcane"]
Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")]
Coordinate = Annotated[StrictInt, Field(ge=0, le=127)]
Extent = Annotated[StrictInt, Field(ge=1, le=128)]
Position = Annotated[list[Coordinate], Field(min_length=2, max_length=2)]
Size = Annotated[list[Extent], Field(min_length=2, max_length=2)]


class TokenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class MapSpec(TokenModel):
    width: Annotated[StrictInt, Field(ge=8, le=128)] = 32
    height: Annotated[StrictInt, Field(ge=8, le=128)] = 24
    biome: Biome = "forest"


class ObjectAttributes(TokenModel):
    size: Size = Field(default_factory=lambda: [1, 1])
    style: Annotated[str, Field(max_length=80)] | None = None
    material: Annotated[str, Field(max_length=80)] | None = None
    tags: Annotated[list[Annotated[str, Field(max_length=40)]], Field(max_length=12)] = Field(
        default_factory=list
    )


class ObjectToken(TokenModel):
    object_type: ObjectType
    id: Identifier
    position: Position
    attributes: ObjectAttributes = Field(default_factory=ObjectAttributes)


class RelationToken(TokenModel):
    relation_type: Literal["near", "inside", "connected_to"]
    source: Identifier
    target: Identifier


class WorldState(TokenModel):
    schema_version: Literal["1.0"] = "1.0"
    scene: Identifier
    map: MapSpec = Field(default_factory=MapSpec)
    objects: Annotated[list[ObjectToken], Field(min_length=1, max_length=256)]
    relations: Annotated[list[RelationToken], Field(max_length=512)] = Field(default_factory=list)


def strict_json_schema() -> dict:
    """Adapt Pydantic defaults to the required-fields subset used by OpenAI."""
    schema = WorldState.model_json_schema()

    def visit(node: object) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema
