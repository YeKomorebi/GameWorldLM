import json
from collections import Counter

import pytest
from PIL import Image

from assets.catalog import VISUALS
from engine.pygame_renderer import MAP_TOP, MARGIN, PygameRenderer, export_placeholder_assets
from examples.scenarios import load_scenarios
from gameworldlm.cli import main
from llm.fixture_backend import FixtureBackend
from llm.providers import LLMError
from world.generator import WorldGenerator
from world.io import load_world, save_world


def _pixel_counts(image):
    return {color: count for count, color in image.getcolors(image.width * image.height)}


def test_five_scenario_cli_end_to_end(tmp_path):
    assert main(["examples", "--provider", "fixture", "--output-dir", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "fixture"
    assert len(manifest["scenarios"]) == 5
    assert all(item["status"] == "ok" for item in manifest["scenarios"])
    for scenario in load_scenarios():
        world = load_world(tmp_path / f"{scenario.name}.json")
        actual = Counter(obj.object_type for obj in world.objects)
        assert all(actual[kind] == count for kind, count in scenario.expected_counts.items())
        with Image.open(tmp_path / f"{scenario.name}.png") as image:
            assert image.format == "PNG"
            pixels = _pixel_counts(image.crop((MARGIN, MAP_TOP, MARGIN + 768, MAP_TOP + 576)))
            for kind in actual:
                assert pixels[VISUALS[kind].color] > 0, f"Missing rendered {kind}"


def test_render_is_deterministic_and_places_objects_at_coordinates(forest, tmp_path):
    renderer = PygameRenderer()
    first = renderer.render(forest, tmp_path / "first.png")
    second = renderer.render(forest, tmp_path / "second.png")
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        river = image.crop(
            (MARGIN + 17 * 24 + 2, MAP_TOP + 24, MARGIN + 19 * 24 - 2, MAP_TOP + 240)
        )
        assert _pixel_counts(river)[VISUALS["river"].color] > 100


def test_assets_are_real_nonempty_pngs(tmp_path):
    paths = export_placeholder_assets(tmp_path)
    assert len(paths) == len(VISUALS)
    for path in paths:
        with Image.open(path) as image:
            assert image.size == (32, 32)
            assert image.getbbox() is not None


def test_validate_and_render_standalone(forest, tmp_path):
    world_path = save_world(forest, tmp_path / "world.json")
    assert main(["validate", str(world_path)]) == 0
    assert main(["render", str(world_path), "--output", str(tmp_path / "map.png")]) == 0
    assert main(["validate", str(tmp_path / "missing.json")]) == 1


def test_fixture_never_invents_a_world_for_unknown_prompt():
    with pytest.raises(LLMError, match="exact example prompts"):
        WorldGenerator(FixtureBackend()).generate("A completely new world")


@pytest.mark.parametrize("tile_size", [0, 15, 49, True, 24.5])
def test_bad_tile_sizes_are_rejected(tile_size):
    with pytest.raises(ValueError):
        PygameRenderer(tile_size)
