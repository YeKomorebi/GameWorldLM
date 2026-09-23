"""Validate spatial meaning after schema validation succeeds."""

from dataclasses import dataclass
from itertools import combinations

from world.geometry import bounds, contains, layer, overlaps, shares_edge, tile_distance
from world.schema import WorldState


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


class WorldValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        super().__init__("; ".join(f"{issue.code}: {issue.message}" for issue in issues))


def validate_world(world: WorldState) -> WorldState:
    issues: list[ValidationIssue] = []
    by_id = {}
    for obj in world.objects:
        if obj.id in by_id:
            issues.append(ValidationIssue("duplicate_id", f"Duplicate object ID: {obj.id}"))
        by_id[obj.id] = obj
        _, _, right, bottom = bounds(obj)
        if right > world.map.width or bottom > world.map.height:
            issues.append(
                ValidationIssue("out_of_bounds", f"{obj.id} footprint exceeds map bounds")
            )

    seen = set()
    valid_containment = set()
    for rel in world.relations:
        pair = (rel.source, rel.target)
        canonical_pair = pair if rel.relation_type == "inside" else tuple(sorted(pair))
        key = (rel.relation_type, *canonical_pair)
        if key in seen:
            issues.append(ValidationIssue("duplicate_relation", f"Repeated relation: {key}"))
        seen.add(key)
        if rel.source == rel.target:
            issues.append(ValidationIssue("self_relation", f"{rel.source} relates to itself"))
            continue
        if rel.source not in by_id or rel.target not in by_id:
            issues.append(ValidationIssue("missing_reference", f"Unknown ID in relation: {pair}"))
            continue
        source, target = by_id[rel.source], by_id[rel.target]
        if rel.relation_type == "near" and tile_distance(source, target) > 3:
            issues.append(ValidationIssue("not_near", f"{pair} must be within 3 occupied tiles"))
        elif rel.relation_type == "inside":
            if not contains(target, source) or bounds(target) == bounds(source):
                issues.append(
                    ValidationIssue("not_inside", f"{rel.source} must fit within {rel.target}")
                )
            else:
                valid_containment.add(frozenset(pair))
        elif rel.relation_type == "connected_to" and not shares_edge(source, target):
            issues.append(ValidationIssue("not_connected", f"{pair} must share a footprint edge"))

    # Terrain and entities occupy separate layers; explicit containment permits nesting.
    for a, b in combinations(world.objects, 2):
        if (
            layer(a) == layer(b)
            and overlaps(a, b)
            and frozenset((a.id, b.id)) not in valid_containment
        ):
            issues.append(ValidationIssue("overlap", f"Same-layer overlap: {a.id}, {b.id}"))
        elif (
            {layer(a), layer(b)} == {1, 2}
            and overlaps(a, b)
            and frozenset((a.id, b.id)) not in valid_containment
            and a.object_type != "bridge"
            and b.object_type != "bridge"
        ):
            issues.append(
                ValidationIssue("blocked_entity", f"Entity intersects structure: {a.id}, {b.id}")
            )

    if issues:
        raise WorldValidationError(issues)
    return world
