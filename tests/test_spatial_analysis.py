import json
from copy import deepcopy

import pytest

from evaluation.spatial import analyze, relation_scores, write_analysis
from world.schema import WorldState


def object_row(kind, object_id, x, y, width=1, height=1):
    return {
        "object_type": kind,
        "id": object_id,
        "position": [x, y],
        "attributes": {"size": [width, height]},
    }


def world_row(objects=None, relations=None):
    return WorldState(
        scene="test",
        map={"width": 16, "height": 16, "biome": "forest"},
        objects=objects
        or [object_row("house", "house_a", 2, 2, 3, 3), object_row("npc", "npc_a", 5, 2)],
        relations=relations or [],
    )


def edge(kind="near", source="npc_a", target="house_a"):
    return {"relation_type": kind, "source": source, "target": target}


def sample(world, minimum=None, required=None):
    return {
        "id": "sample",
        "messages": [
            {"role": "system", "content": "World JSON"},
            {"role": "user", "content": "A house and npc"},
            {"role": "assistant", "content": world.model_dump_json()},
        ],
        "expectations": {
            "object_counts": {"house": 1, "npc": 1},
            "minimum_relations": minimum or {},
        },
        "required_relations": required or [],
    }


def score(world, **kwargs):
    return analyze(
        [sample(world, **kwargs)], [{"id": "sample", "raw_response": world.model_dump_json()}]
    )


@pytest.mark.parametrize("raw", ["```json\n{}\n```", "{", '{"x":NaN}', "[]"])
def test_invalid_outputs_cannot_receive_spatial_credit(raw):
    world = world_row()
    result = analyze([sample(world, {"near": 1})], [{"id": "sample", "raw_response": raw}])
    assert result["metrics"]["near_accuracy"] == 0
    assert result["metrics"]["boundary_accuracy"] == 0
    assert result["metrics"]["overlap_free_rate"] == 0
    assert result["failure_sample_counts"]["other"] == 1
    assert result["details"][0]["raw_response"] == raw
    assert result["details"][0]["failure_reasons"][0]["message"]


def test_missing_required_edge_and_unrelated_edge_are_penalized():
    world = world_row(relations=[edge()])
    spec = sample(world, {"near": 1}, [edge(target="house_b")])
    result = relation_scores(world, spec)["near"]
    assert (result["passed"], result["total"]) == (1, 2)
    assert result["missing_slots"] == 1
    assert score(world_row(), minimum={"near": 1})["metrics"]["near_accuracy"] == 0
    assert score(world_row())["metrics"]["near_accuracy"] is None


def test_symmetric_duplicates_and_invalid_references():
    world = world_row(
        relations=[
            edge(),
            edge(source="house_a", target="npc_a"),
            edge(source="missing"),
            edge(source="house_a"),
        ]
    )
    result = score(world)
    assert result["relation_totals"]["near"] == {"passed": 1, "total": 4}
    assert result["validator_code_counts"] == {
        "duplicate_relation": 1,
        "missing_reference": 1,
        "self_relation": 1,
    }
    assert result["failure_sample_counts"]["invalid_reference"] == 1


@pytest.mark.parametrize("position,expected", [([7, 2], 1), ([8, 2], 0)])
def test_near_uses_occupied_tile_chebyshev_distance(position, expected):
    world = world_row(relations=[edge()])
    world.objects[1].position = position
    assert score(world)["metrics"]["near_accuracy"] == expected


@pytest.mark.parametrize("position,expected", [([5, 2], 1), ([5, 5], 0), ([4, 2], 0)])
def test_connection_requires_positive_shared_edge(position, expected):
    world = world_row(relations=[edge("connected_to")])
    world.objects[1].position = position
    assert score(world)["metrics"]["connected_to_accuracy"] == expected


def test_containment_is_directional_and_does_not_penalize_legal_overlap():
    world = world_row(relations=[edge("inside")])
    world.objects[1].position = [3, 3]
    result = score(world)
    assert result["metrics"]["inside_accuracy"] == 1
    assert result["metrics"]["overlap_free_rate"] == 1
    reversed_world = world.model_copy(deep=True)
    reversed_world.relations[0].source, reversed_world.relations[0].target = "house_a", "npc_a"
    assert score(reversed_world)["metrics"]["inside_accuracy"] == 0
    world.objects[1].position = [2, 2]
    world.objects[1].attributes.size = [3, 3]
    assert score(world)["metrics"]["inside_accuracy"] == 0


def test_blocked_entities_keep_exact_reason_and_independent_object_counts(tmp_path):
    world = world_row()
    world.objects[1].position = [2, 2]
    result = score(world)
    assert result["metrics"]["object_count_accuracy"] == 1
    assert result["metrics"]["overlap_free_rate"] == 0
    assert result["failure_issue_counts"]["overlap"] == 1
    reason = result["details"][0]["failure_reasons"][0]
    assert reason == {
        "code": "blocked_entity",
        "category": "overlap",
        "message": "Entity intersects structure: house_a, npc_a",
    }
    write_analysis(tmp_path, {"lora": result})
    saved = json.loads((tmp_path / "lora_spatial_samples.jsonl").read_text())
    assert saved["prompt"] == "A house and npc"
    assert saved["generated_json"]["objects"][1]["position"] == [2, 2]
    assert saved["validator_result"]["issues"] == [reason]


def test_boundary_checks_extent_and_invalid_ids_cannot_earn_credit():
    world = world_row()
    world.objects[0].position = [14, 2]
    assert score(world)["failure_issue_counts"]["out_of_bounds"] == 1
    assert score(world)["metrics"]["boundary_accuracy"] == 0
    world.objects[0].id = "npc_a"
    result = score(world)
    assert result["metrics"]["overlap_free_rate"] == 0


def test_micro_denominators_include_parse_errors_and_omit_unrequested_empty_types():
    world = world_row(relations=[edge()])
    a = sample(world, {"near": 1})
    b = deepcopy(a) | {"id": "second", "expectations": {"minimum_relations": {"near": 3}}}
    result = analyze(
        [a, b],
        [
            {"id": "sample", "raw_response": world.model_dump_json()},
            {"id": "second", "raw_response": "invalid"},
        ],
    )
    assert result["metrics"]["near_accuracy"] == 0.25
    assert result["metrics"]["inside_accuracy"] is None
    assert result["metrics"]["json_parse_rate"] == 0.5


def test_prediction_identity_mismatch_rejected():
    with pytest.raises(ValueError, match="exactly"):
        analyze([sample(world_row())], [])


def test_inference_errors_are_preserved_without_attempting_parse():
    result = analyze([sample(world_row())], [{"id": "sample", "error": "GPU failure"}])
    assert result["details"][0]["failure_reasons"][0]["message"] == "GPU failure"
    assert not result["details"][0]["validator_result"]["evaluated"]


def test_overflowing_json_numbers_do_not_break_diagnostic_export(tmp_path):
    raw = '{"extra":1e999}'
    result = analyze([sample(world_row())], [{"id": "sample", "raw_response": raw}])
    write_analysis(tmp_path, {"lora": result})
    saved = json.loads((tmp_path / "lora_spatial_samples.jsonl").read_text())
    assert saved["raw_response"] == raw
    assert saved["decoded_json_error"]
    assert not saved["schema_valid"]
