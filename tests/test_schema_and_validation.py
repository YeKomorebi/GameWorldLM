from typing import get_args

import pytest
from pydantic import ValidationError

from assets.catalog import BIOME_COLORS, VISUALS
from examples.scenarios import fixture_worlds
from world.geometry import tile_distance
from world.schema import (
    Biome,
    ObjectToken,
    ObjectType,
    RelationToken,
    WorldState,
    strict_json_schema,
)
from world.validator import WorldValidationError, validate_world


@pytest.mark.parametrize("scene", list(fixture_worlds()))
def test_fixtures_validate_and_roundtrip(scene):
    world = fixture_worlds()[scene]
    assert validate_world(world) == WorldState.model_validate_json(world.model_dump_json())


@pytest.mark.parametrize("coordinate", ["2", 2.5, True, -1, 128])
def test_coordinates_are_bounded_strict_integers(coordinate):
    with pytest.raises(ValidationError):
        ObjectToken(object_type="house", id="house_001", position=[coordinate, 0])


@pytest.mark.parametrize(
    "change",
    [
        {"object_type": "unsupported"},
        {"id": "../escape"},
        {"surprise": 1},
        {"position": [1]},
        {"attributes": {"size": [0, 2]}},
    ],
)
def test_object_contract_rejects_invalid_data(change):
    data = {"object_type": "house", "id": "house_001", "position": [0, 0]}
    with pytest.raises(ValidationError):
        ObjectToken.model_validate(data | change)


def test_all_token_types_have_visuals():
    assert set(get_args(ObjectType)) == set(VISUALS)
    assert set(get_args(Biome)) == set(BIOME_COLORS)


def test_strict_schema_has_no_defaults_and_requires_all_properties():
    def visit(node):
        if isinstance(node, dict):
            assert "default" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(strict_json_schema())


def _pair():
    return WorldState(
        scene="pair",
        objects=[
            ObjectToken(
                object_type="house", id="house_a", position=[1, 1], attributes={"size": [3, 3]}
            ),
            ObjectToken(
                object_type="house", id="house_b", position=[6, 1], attributes={"size": [2, 2]}
            ),
        ],
    )


@pytest.mark.parametrize(
    "case,code",
    [
        ("duplicate", "duplicate_id"),
        ("bounds", "out_of_bounds"),
        ("overlap", "overlap"),
        ("missing", "missing_reference"),
        ("self", "self_relation"),
        ("far", "not_near"),
        ("inside", "not_inside"),
        ("disconnected", "not_connected"),
        ("reversed", "duplicate_relation"),
        ("entity", "blocked_entity"),
    ],
)
def test_invalid_spatial_cases(case, code):
    world = _pair()
    a, b = world.objects
    if case == "duplicate":
        b.id = a.id
    elif case == "bounds":
        b.position = [31, 23]
    elif case == "overlap":
        b.position = [2, 2]
    elif case == "entity":
        b.object_type = "npc"
        b.position = [2, 2]
    else:
        rel = RelationToken(relation_type="near", source=a.id, target=b.id)
        if case == "missing":
            rel.target = "unknown"
        elif case == "self":
            rel.target = a.id
        elif case == "far":
            b.position = [20, 20]
        elif case == "inside":
            rel.relation_type = "inside"
        elif case == "disconnected":
            rel.relation_type = "connected_to"
        elif case == "reversed":
            world.relations.append(RelationToken(relation_type="near", source=b.id, target=a.id))
        world.relations.append(rel)
    with pytest.raises(WorldValidationError) as exc:
        validate_world(world)
    assert code in {issue.code for issue in exc.value.issues}


def test_edge_connection_and_near_use_full_footprint():
    world = _pair()
    a, b = world.objects
    assert tile_distance(a, b) == 3
    world.relations = [RelationToken(relation_type="near", source=a.id, target=b.id)]
    validate_world(world)
    b.position = [4, 1]
    world.relations[0].relation_type = "connected_to"
    validate_world(world)
    b.position = [4, 4]
    with pytest.raises(WorldValidationError, match="not_connected"):
        validate_world(world)


def test_valid_inside_permits_nested_same_layer_objects():
    world = _pair()
    a, b = world.objects
    a.attributes.size = [8, 8]
    b.position = [3, 3]
    world.relations = [RelationToken(relation_type="inside", source=b.id, target=a.id)]
    validate_world(world)


def test_identical_footprints_do_not_allow_inside_cycles():
    world = _pair()
    a, b = world.objects
    b.position = a.position.copy()
    b.attributes.size = a.attributes.size.copy()
    world.relations = [RelationToken(relation_type="inside", source=b.id, target=a.id)]
    with pytest.raises(WorldValidationError, match="not_inside"):
        validate_world(world)
