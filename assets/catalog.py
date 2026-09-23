"""Object appearance is kept out of world tokens and model prompts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectVisual:
    color: tuple[int, int, int]
    glyph: str


VISUALS = {
    "house": ObjectVisual((211, 131, 93), "building"),
    "tree": ObjectVisual((57, 137, 87), "tree"),
    "river": ObjectVisual((61, 154, 203), "water"),
    "road": ObjectVisual((185, 176, 145), "path"),
    "bridge": ObjectVisual((151, 112, 76), "bridge"),
    "monster": ObjectVisual((213, 78, 101), "entity"),
    "castle": ObjectVisual((151, 173, 194), "fort"),
    "ruin": ObjectVisual((158, 150, 120), "ruin"),
    "rock": ObjectVisual((120, 132, 135), "rock"),
    "dune": ObjectVisual((223, 192, 116), "path"),
    "tower": ObjectVisual((152, 131, 187), "fort"),
    "wall": ObjectVisual((107, 121, 143), "fort"),
    "library": ObjectVisual((185, 108, 121), "building"),
    "portal": ObjectVisual((174, 112, 222), "portal"),
    "npc": ObjectVisual((237, 199, 85), "entity"),
    "crystal": ObjectVisual((77, 209, 196), "crystal"),
    "courtyard": ObjectVisual((159, 181, 165), "path"),
    "vehicle": ObjectVisual((192, 177, 86), "vehicle"),
}

BIOME_COLORS = {
    "forest": (45, 69, 58),
    "snow": (216, 229, 233),
    "desert": (221, 204, 156),
    "city": (86, 93, 98),
    "arcane": (72, 94, 91),
}
