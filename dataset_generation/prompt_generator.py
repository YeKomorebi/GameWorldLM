"""Deterministic scene diversity, with themes kept outside the WorldState schema."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict

from world.evaluation import GenerationExpectations
from world.schema import Identifier, ObjectType

PROMPT_VERSION = "0.3.0"
Theme = Literal["forest", "desert", "ice", "city", "dungeon", "fantasy", "cyberpunk"]
THEME_BIOMES = {
    "forest": "forest",
    "desert": "desert",
    "ice": "snow",
    "city": "city",
    "dungeon": "arcane",
    "fantasy": "arcane",
    "cyberpunk": "city",
}
THEMES = tuple(THEME_BIOMES)
PROFILES = {
    "forest": (
        (
            "woodland village",
            "ranger outpost",
            "forest sanctuary",
            "logging settlement",
            "hidden refuge",
            "border hamlet",
            "herbalist retreat",
            "abandoned camp",
        ),
        ("house", "tree", "rock", "npc", "monster", "road"),
        ("moss-covered timber", "living wood", "weathered oak", "rough cedar"),
    ),
    "desert": (
        (
            "oasis outpost",
            "buried ruins",
            "caravan settlement",
            "desert observatory",
            "sandstone refuge",
            "excavation camp",
            "lost trading post",
            "dune fortress",
        ),
        ("ruin", "dune", "rock", "npc", "monster", "road"),
        ("carved sandstone", "sun-bleached stone", "cracked clay", "copper-trimmed stone"),
    ),
    "ice": (
        (
            "glacier fortress",
            "frozen citadel",
            "polar refuge",
            "winter watchpost",
            "snowbound keep",
            "frost sanctuary",
            "icebound stronghold",
            "northern court",
        ),
        ("castle", "tree", "crystal", "npc", "monster", "road"),
        ("frosted stone", "blue ice", "snow-covered granite", "silver-trimmed ice"),
    ),
    "city": (
        (
            "abandoned district",
            "market neighborhood",
            "industrial quarter",
            "walled suburb",
            "evacuation zone",
            "reclaimed city block",
            "old town",
            "survivor settlement",
        ),
        ("house", "ruin", "vehicle", "npc", "monster", "road"),
        ("weathered brick", "reinforced concrete", "reclaimed metal", "painted masonry"),
    ),
    "dungeon": (
        (
            "underground prison",
            "forgotten crypt",
            "sealed vault",
            "subterranean archive",
            "ritual chamber",
            "ancient catacomb",
            "cavern stronghold",
            "buried temple",
        ),
        ("ruin", "wall", "crystal", "npc", "monster", "road"),
        ("dark basalt", "chiseled limestone", "rune-covered stone", "corroded iron"),
    ),
    "fantasy": (
        (
            "magic academy",
            "wizard enclave",
            "enchanted court",
            "arcane sanctuary",
            "spell research camp",
            "royal mage outpost",
            "crystal monastery",
            "portal refuge",
        ),
        ("tower", "library", "portal", "npc", "monster", "road"),
        ("luminous marble", "carved crystal", "ivy-covered stone", "gold-trimmed quartz"),
    ),
    "cyberpunk": (
        (
            "neon district",
            "corporate enclave",
            "hacker refuge",
            "drone depot",
            "augmented settlement",
            "data market",
            "abandoned tech campus",
            "security outpost",
        ),
        ("tower", "house", "vehicle", "npc", "monster", "road"),
        ("neon-trimmed steel", "chrome and glass", "recycled electronics", "armored alloy"),
    ),
}
RELATIONS = {
    "open": "Leave space between buildings. No relations are required.",
    "near": "Place one npc near one {primary}; include that near relation (distance <=3 tiles).",
    "inside": (
        "Place one npc fully inside one {primary}, with unequal footprints; "
        "include that inside relation. Keep other entities outside structures."
    ),
    "connected_to": (
        "Make the road share a footprint edge with one {primary}; "
        "include that connected_to relation. Edge contact must have nonzero length."
    ),
}


class PromptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: Identifier
    theme: Theme
    prompt: str
    expectations: GenerationExpectations
    descriptors: dict[str, str]


def generate_prompts(count: int = 10_000, seed: int = 42) -> list[PromptSpec]:
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 100_000:
        raise ValueError("count must be an integer in [1, 100000]")
    rng = random.Random(seed)
    cases, seen = [], set()
    for index in range(count):
        theme = THEMES[index % len(THEMES)]
        settings, kinds, styles = PROFILES[theme]
        while True:
            setting, style = rng.choice(settings), rng.choice(styles)
            mood = rng.choice(("peaceful", "ominous", "festive", "desolate", "mysterious"))
            layout = rng.choice(
                (
                    "central clearing",
                    "eastern square",
                    "western square",
                    "northern gathering area",
                    "southern gathering area",
                )
            )
            width, height = rng.choice(((32, 24), (36, 28), (40, 32), (48, 32)))
            relation = rng.choice(tuple(RELATIONS))
            counts = {kind: rng.randint(1, 4) for kind in kinds}
            counts["road"] = 1
            request = ", ".join(f"{number} {kind}" for kind, number in counts.items())
            prompt = (
                f"Create a {mood} {setting} in a {theme} game world. "
                f"Use a {width} by {height} tile map with biome '{THEME_BIOMES[theme]}'. "
                f"Include exactly these objects and no others: {request}. "
                f"Use {style} architecture, recorded in the appropriate style/material attributes. "
                f"Arrange the scene around a {layout}, keeping every footprint within the map. "
                + RELATIONS[relation].format(primary=kinds[0])
                + " Use compact footprints and leave open tiles for movement. "
                "Represent the theme using supported object types and attributes only."
            )
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if digest not in seen:
                seen.add(digest)
                break
        cases.append(
            PromptSpec(
                id=f"{theme}_{index + 1:05d}_{digest[:10]}",
                theme=theme,
                prompt=prompt,
                expectations=GenerationExpectations(
                    object_counts={kind: counts.get(kind, 0) for kind in get_args(ObjectType)},
                    map={"width": width, "height": height, "biome": THEME_BIOMES[theme]},
                    minimum_relations={} if relation == "open" else {relation: 1},
                ),
                descriptors={
                    "setting": setting,
                    "style": style,
                    "mood": mood,
                    "layout": layout,
                    "relation": relation,
                },
            )
        )
    return cases


def write_prompts(path: Path, cases: list[PromptSpec]) -> str:
    """Write a stable inventory once; refuse to silently replace a different experiment."""
    content = "".join(case.model_dump_json() + "\n" for case in cases).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(
                "Prompt inventory differs; use the original count/seed or a new directory"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    return digest


def inventory_summary(cases: list[PromptSpec]) -> dict:
    return {
        "total": len(cases),
        "themes": dict(Counter(case.theme for case in cases)),
        "relations": dict(Counter(case.descriptors["relation"] for case in cases)),
        "unique_prompts": len({case.prompt for case in cases}),
        "theme_biomes": THEME_BIOMES,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate balanced game scene instructions offline"
    )
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("dataset/prompts.jsonl"))
    args = parser.parse_args(argv)
    try:
        cases = generate_prompts(args.count, args.seed)
        digest = write_prompts(args.output, cases)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Prompt generation failed: {exc}\n")
    print(json.dumps(inventory_summary(cases) | {"sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
