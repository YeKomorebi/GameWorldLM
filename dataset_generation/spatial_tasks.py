"""Construct bounded spatial constraint tasks with independently validated witnesses.

Witnesses are synthetic labels, not LLM completions. Benchmark task templates and
curriculum templates differ; the shared placement search only proves feasibility.
"""

import hashlib
import json
import random
from collections import Counter
from itertools import product
from typing import get_args

from world.geometry import bounds, contains, shares_edge, tile_distance
from world.schema import MapSpec, ObjectToken, ObjectType, RelationToken, WorldState
from world.validator import WorldValidationError, validate_world

from .prompt_generator import THEME_BIOMES, THEMES

RELATIONS = ("near", "inside", "connected_to")
DIFFICULTIES = ("easy", "medium", "hard")
PRIMARY = {
    "forest": "house",
    "desert": "ruin",
    "ice": "castle",
    "city": "house",
    "dungeon": "ruin",
    "fantasy": "library",
    "cyberpunk": "tower",
}


def edge_valid(kind: str, source: ObjectToken, target: ObjectToken) -> bool:
    if kind == "near":
        return tile_distance(source, target) <= 3
    if kind == "inside":
        return contains(target, source) and bounds(target) != bounds(source)
    return shares_edge(source, target)


def place_objects(
    specs: list[dict], relations: list[dict], map_spec: MapSpec, rng: random.Random, scene: str
) -> WorldState:
    placed, visited = [], 0
    tokens = [RelationToken.model_validate(rel) for rel in relations]

    def search(index: int):
        nonlocal visited
        if index == len(specs):
            return WorldState(scene=scene, map=map_spec, objects=placed.copy(), relations=tokens)
        spec = specs[index]
        width, height = spec["size"]
        candidates = list(
            product(range(map_spec.width - width + 1), range(map_spec.height - height + 1))
        )
        rng.shuffle(candidates)
        # Hard tasks include boundary contact, still using the exact footprint rules.
        if spec.get("boundary"):
            candidates = [
                (x, y)
                for x, y in candidates
                if x in (0, map_spec.width - width) or y in (0, map_spec.height - height)
            ]
        by_id = {obj.id: obj for obj in placed}
        active = [
            rel
            for rel in tokens
            if (rel.source == spec["id"] and rel.target in by_id)
            or (rel.target == spec["id"] and rel.source in by_id)
        ]
        for x, y in candidates:
            obj = ObjectToken(
                object_type=spec["object_type"],
                id=spec["id"],
                position=[x, y],
                attributes={"size": spec["size"]},
            )
            if any(
                not edge_valid(
                    rel.relation_type,
                    obj if rel.source == obj.id else by_id[rel.source],
                    obj if rel.target == obj.id else by_id[rel.target],
                )
                for rel in active
            ):
                continue
            visited += 1
            if visited > 20_000:
                raise ValueError("Placement search exhausted its bounded budget")
            ids = set(by_id) | {obj.id}
            trial = WorldState(
                scene=scene,
                map=map_spec,
                objects=placed + [obj],
                relations=[rel for rel in tokens if rel.source in ids and rel.target in ids],
            )
            try:
                validate_world(trial)
            except WorldValidationError:
                continue
            placed.append(obj)
            result = search(index + 1)
            if result is not None:
                return result
            placed.pop()
        return None

    result = search(0)
    if result is None:
        raise ValueError("No feasible spatial placement found")
    return validate_world(result)


def make_task(
    index: int, relation: str, difficulty: str, seed: int, *, benchmark: bool = False
) -> tuple[dict, WorldState]:
    if relation not in RELATIONS or difficulty not in DIFFICULTIES:
        raise ValueError("Unknown relation or difficulty")
    domain = "benchmark" if benchmark else "curriculum"
    task_seed = int(
        hashlib.sha256(f"{domain}:{seed}:{index}:{relation}:{difficulty}".encode()).hexdigest()[
            :16
        ],
        16,
    )
    rng = random.Random(task_seed)
    theme = THEMES[index % len(THEMES)]
    level = DIFFICULTIES.index(difficulty)
    size = rng.choice((24, 28, 32)) + level * 8
    map_spec = MapSpec(width=size, height=size - rng.choice((0, 4)), biome=THEME_BIOMES[theme])
    specs, edges = [], []

    def add(kind, extent, boundary=False):
        object_id = f"{kind}_{1 + sum(obj['object_type'] == kind for obj in specs):03d}"
        specs.append({"id": object_id, "object_type": kind, "size": extent, "boundary": boundary})
        return object_id

    def connect(kind, source, target):
        edges.append({"relation_type": kind, "source": source, "target": target})

    primary = PRIMARY[theme]
    pair_count = level + 1
    if benchmark:
        # Held-out multi-edge stars/chains contrast with curriculum disjoint pairs.
        anchor = add(primary, [rng.randint(5, 8), rng.randint(5, 8)], boundary=level == 2)
        for _ in range(pair_count):
            kind = rng.choice(("npc", "monster")) if relation != "connected_to" else "road"
            child = add(kind, [rng.randint(1, 2), rng.randint(1, 2)])
            connect(relation, child, anchor)
            if relation == "connected_to":
                anchor = child
        if level == 2:
            extra = add("crystal", [2, 2])
            connect("near", extra, specs[0]["id"])
            if relation != "inside":
                resident = add("npc", [1, 1])
                connect("inside", resident, specs[0]["id"])
            else:
                entrance = add("road", [2, 1])
                connect("connected_to", entrance, specs[0]["id"])
    else:
        kinds = [relation] * pair_count
        if level == 2:
            kinds += [kind for kind in RELATIONS if kind != relation]
        for i, kind in enumerate(kinds):
            anchor = add(
                primary, [rng.randint(3, 7), rng.randint(3, 7)], boundary=level == 2 and i == 0
            )
            child_kind = rng.choice(("npc", "monster")) if kind != "connected_to" else "road"
            child = add(child_kind, [rng.randint(1, 2), rng.randint(1, 2)])
            connect(kind, child, anchor)
    for _ in range(level + (1 if benchmark else 0)):
        add(rng.choice(("tree", "rock", "crystal")), [rng.randint(1, 3), rng.randint(1, 3)])
    scene = f"{domain}_{relation}_{difficulty}_{index:04d}"
    world = place_objects(specs, edges, map_spec, rng, scene)
    counts = Counter(obj["object_type"] for obj in specs)
    objects_text = "; ".join(
        f"{obj['id']} ({obj['object_type']}, {obj['size'][0]}x{obj['size'][1]})" for obj in specs
    )
    relation_text = "; ".join(
        f"{rel['source']} {rel['relation_type']} {rel['target']}" for rel in edges
    )
    boundary = next((obj["id"] for obj in specs if obj["boundary"]), None)
    if benchmark:
        prompt = (
            f"Design a {theme} expedition site on a {map_spec.width} by {map_spec.height} "
            f"tile map (biome {map_spec.biome}). The inventory is exactly: {objects_text}. "
            f"Spatial requirements: {relation_text}. Return these IDs and footprints, "
            "with an explicit relation token for every requirement."
        )
    else:
        prompt = (
            f"Create a {theme} settlement, biome {map_spec.biome}, map "
            f"{map_spec.width}x{map_spec.height}. Use exactly these objects, IDs and sizes: "
            f"{objects_text}. Include and satisfy: {relation_text}."
        )
    if boundary:
        prompt += f" Place {boundary} flush with one map boundary."
    prompt += (
        " Keep every footprint in bounds. Avoid same-layer overlaps and blocked entities; "
        "explicit valid containment is allowed. near uses occupied-tile distance <=3; "
        "inside requires full containment with unequal footprints; connected_to requires "
        "a shared edge of positive length. Return only World JSON."
    )
    row = {
        "id": scene,
        "prompt": prompt,
        "theme": theme,
        "difficulty": difficulty,
        "focus_relation": relation,
        "seed": seed,
        "evaluation_only": benchmark,
        "expectations": {
            "object_counts": {kind: counts[kind] for kind in get_args(ObjectType)},
            "map": map_spec.model_dump(),
            "minimum_relations": dict(Counter(rel["relation_type"] for rel in edges)),
        },
        "required_relations": edges,
        "required_objects": [{k: v for k, v in obj.items() if k != "boundary"} for obj in specs],
        "boundary_object": boundary,
        "template_family": "expedition_stars_chains" if benchmark else "settlement_disjoint_pairs",
    }
    return row, world


def check_task(world: WorldState, row: dict) -> list[str]:
    """Check prompt-specific IDs/sizes/edges in addition to the unmodified validator."""
    from evaluation.spatial import relation_key

    errors = []
    objects = {obj.id: obj for obj in world.objects}
    required = row.get("required_objects", [])
    if required and set(objects) != {obj["id"] for obj in required}:
        errors.append("Object IDs differ from the requested inventory")
    for spec in required:
        obj = objects.get(spec["id"])
        if (
            obj is None
            or obj.object_type != spec["object_type"]
            or obj.attributes.size != spec["size"]
        ):
            errors.append(f"Wrong or missing object/footprint: {spec['id']}")
    predicted = {relation_key(rel) for rel in world.relations}
    for rel in row.get("required_relations", []):
        if relation_key(rel) not in predicted:
            errors.append(f"Missing required edge: {relation_key(rel)}")
    boundary = objects.get(row.get("boundary_object"))
    if row.get("boundary_object") and (
        boundary is None
        or not (
            bounds(boundary)[0] == 0
            or bounds(boundary)[1] == 0
            or bounds(boundary)[2] == world.map.width
            or bounds(boundary)[3] == world.map.height
        )
    ):
        errors.append("Requested boundary contact is missing")
    return errors


def world_fingerprint(world: WorldState) -> str:
    payload = world.model_dump(exclude={"scene"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
