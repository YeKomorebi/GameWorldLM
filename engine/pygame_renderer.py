"""Headless pygame raster rendering; no display, audio, or game loop required."""

import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from assets.catalog import BIOME_COLORS, VISUALS  # noqa: E402
from world.geometry import layer  # noqa: E402
from world.schema import WorldState  # noqa: E402
from world.validator import validate_world  # noqa: E402

INK = (30, 39, 42)
PAPER = (245, 247, 246)
MUTED = (88, 102, 103)
MARGIN = 32
MAP_TOP = 104
SIDEBAR = 342


def make_sprite(object_type: str, size: int = 32) -> pygame.Surface:
    visual = VISUALS[object_type]
    surface = pygame.Surface((32, 32), pygame.SRCALPHA)
    color = visual.color
    dark = tuple(max(0, channel - 45) for channel in color)
    light = tuple(min(255, channel + 40) for channel in color)
    glyph = visual.glyph
    if glyph == "tree":
        pygame.draw.rect(surface, (116, 94, 73), (13, 18, 6, 12))
        pygame.draw.circle(surface, dark, (16, 14), 13)
        pygame.draw.circle(surface, color, (14, 12), 10)
        pygame.draw.circle(surface, light, (11, 8), 4)
    elif glyph in {"building", "fort", "ruin"}:
        pygame.draw.rect(surface, dark, (2, 7, 28, 23))
        pygame.draw.rect(surface, color, (4, 9, 24, 19))
        if glyph == "building":
            pygame.draw.polygon(surface, light, [(1, 12), (16, 1), (31, 12)])
        elif glyph == "fort":
            for x in (3, 13, 23):
                pygame.draw.rect(surface, light, (x, 2, 6, 9))
        else:
            pygame.draw.line(surface, INK, (21, 7), (15, 20), 3)
        pygame.draw.rect(surface, INK, (13, 20, 7, 10))
        pygame.draw.rect(surface, (246, 218, 139), (6, 15, 4, 5))
    elif glyph in {"path", "water", "bridge"}:
        surface.fill(color)
        if glyph == "water":
            for y in (9, 22):
                pygame.draw.line(surface, light, (4, y), (14, y), 2)
                pygame.draw.line(surface, light, (20, y + 3), (28, y + 3), 2)
        elif glyph == "bridge":
            for y in (3, 11, 19, 27):
                pygame.draw.line(surface, dark, (0, y), (32, y), 2)
            pygame.draw.line(surface, light, (3, 0), (3, 32), 3)
            pygame.draw.line(surface, light, (28, 0), (28, 32), 3)
        else:
            pygame.draw.rect(surface, light, (4, 6, 3, 2))
            pygame.draw.rect(surface, dark, (23, 24, 3, 2))
    elif glyph == "entity":
        pygame.draw.circle(surface, dark, (16, 17), 12)
        pygame.draw.circle(surface, color, (16, 15), 10)
        pygame.draw.circle(surface, PAPER, (12, 13), 3)
        pygame.draw.circle(surface, PAPER, (21, 13), 3)
        pygame.draw.circle(surface, INK, (13, 14), 1)
        pygame.draw.circle(surface, INK, (20, 14), 1)
    elif glyph == "portal":
        pygame.draw.ellipse(surface, dark, (3, 1, 26, 30))
        pygame.draw.ellipse(surface, color, (6, 3, 20, 26), 4)
        pygame.draw.ellipse(surface, (111, 233, 216), (11, 8, 10, 17))
    elif glyph in {"crystal", "rock"}:
        pygame.draw.polygon(surface, dark, [(4, 20), (12, 3), (23, 6), (29, 23), (16, 30)])
        pygame.draw.polygon(surface, color, [(6, 19), (12, 3), (23, 6), (17, 27)])
        pygame.draw.line(surface, light, (12, 4), (17, 26), 2)
    else:
        pygame.draw.rect(surface, dark, (4, 2, 24, 28), border_radius=3)
        pygame.draw.rect(surface, color, (7, 4, 18, 24), border_radius=2)
        pygame.draw.rect(surface, (76, 125, 139), (9, 7, 14, 7))
    return pygame.transform.scale(surface, (size, size))


def export_placeholder_assets(output_dir: str | Path) -> list[Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in VISUALS:
        path = destination / f"{name}.png"
        pygame.image.save(make_sprite(name), str(path))
        paths.append(path)
    return paths


def _text(surface, text, position, size=18, color=INK, max_width=None):
    font = pygame.font.Font(None, size)
    if max_width:
        while font.size(text)[0] > max_width and size > 10:
            size -= 1
            font = pygame.font.Font(None, size)
        if font.size(text)[0] > max_width:
            while text and font.size(text + "...")[0] > max_width:
                text = text[:-1]
            text += "..."
    surface.blit(font.render(text, True, color), position)


class PygameRenderer:
    def __init__(self, tile_size: int = 24):
        if (
            isinstance(tile_size, bool)
            or not isinstance(tile_size, int)
            or not 16 <= tile_size <= 48
        ):
            raise ValueError("tile_size must be an integer between 16 and 48")
        self.tile_size = tile_size

    def render(self, world: WorldState, destination: str | Path) -> Path:
        validate_world(world)
        tile = self.tile_size
        map_width, map_height = world.map.width * tile, world.map.height * tile
        kinds = sorted({obj.object_type for obj in world.objects})
        legend_height = 58 + ((len(kinds) + 1) // 2) * 24
        width = MARGIN * 3 + map_width + SIDEBAR
        height = max(
            MAP_TOP + map_height + 72, MAP_TOP + legend_height + 76 + len(world.objects) * 25
        )
        if width * height > 32_000_000:
            raise ValueError("Image would exceed 32 million pixels; reduce tile size or map size")
        pygame.font.init()
        canvas = pygame.Surface((width, height))
        canvas.fill(PAPER)
        _text(canvas, "GameWorldLM", (MARGIN, 24), 30)
        _text(canvas, world.scene, (MARGIN + 208, 30), 24, max_width=width - 290)
        _text(
            canvas,
            f"{world.map.biome.upper()}  /  {world.map.width} x {world.map.height} tiles"
            f"  /  {len(world.objects)} objects  /  {len(world.relations)} relations",
            (MARGIN, 62),
            18,
            MUTED,
        )
        area = pygame.Rect(MARGIN, MAP_TOP, map_width, map_height)
        pygame.draw.rect(canvas, BIOME_COLORS[world.map.biome], area)

        assets_dir = Path(__file__).resolve().parents[1] / "assets" / "placeholder"
        sprites = {}
        for name in kinds:
            path = assets_dir / f"{name}.png"
            sprites[name] = (
                pygame.transform.scale(pygame.image.load(str(path)), (tile, tile))
                if path.exists()
                else make_sprite(name, tile)
            )

        # Draw containing footprints before their contents on each layer.
        ordered = sorted(
            world.objects,
            key=lambda obj: (layer(obj), -obj.attributes.size[0] * obj.attributes.size[1]),
        )
        for obj in ordered:
            x, y = obj.position
            sw, sh = obj.attributes.size
            if layer(obj) == 0 or obj.object_type == "bridge":
                for dx in range(sw):
                    for dy in range(sh):
                        canvas.blit(
                            sprites[obj.object_type],
                            (MARGIN + (x + dx) * tile, MAP_TOP + (y + dy) * tile),
                        )
            else:
                sprite = pygame.transform.scale(sprites[obj.object_type], (sw * tile, sh * tile))
                canvas.blit(sprite, (MARGIN + x * tile, MAP_TOP + y * tile))
            outline = pygame.Rect(MARGIN + x * tile, MAP_TOP + y * tile, sw * tile, sh * tile)
            pygame.draw.rect(canvas, VISUALS[obj.object_type].color, outline, 2)

        grid = pygame.Surface((map_width + 1, map_height + 1), pygame.SRCALPHA)
        for x in range(world.map.width + 1):
            pygame.draw.line(grid, (20, 30, 31, 48), (x * tile, 0), (x * tile, map_height))
        for y in range(world.map.height + 1):
            pygame.draw.line(grid, (20, 30, 31, 48), (0, y * tile), (map_width, y * tile))
        canvas.blit(grid, (MARGIN, MAP_TOP))
        pygame.draw.rect(canvas, MUTED, area, 1)

        for x in range(0, world.map.width, 4):
            _text(canvas, str(x), (MARGIN + x * tile + 3, MAP_TOP - 19), 16, MUTED)
        for y in range(0, world.map.height, 4):
            _text(canvas, str(y), (6, MAP_TOP + y * tile + 5), 16, MUTED)
        for index, obj in enumerate(world.objects, 1):
            x, y = obj.position
            marker = pygame.Rect(MARGIN + x * tile + 1, MAP_TOP + y * tile + 1, tile - 2, 15)
            pygame.draw.rect(canvas, INK, marker, border_radius=2)
            _text(canvas, str(index), (marker.x + 2, marker.y + 1), 16, PAPER, marker.width - 4)

        sx = MARGIN * 2 + map_width
        _text(canvas, "OBJECT PALETTE", (sx, MAP_TOP), 20)
        for i, kind in enumerate(kinds):
            x, y = sx + (i % 2) * 163, MAP_TOP + 30 + (i // 2) * 24
            pygame.draw.rect(canvas, VISUALS[kind].color, (x, y, 13, 13))
            _text(canvas, kind, (x + 22, y), 18)
        row_top = MAP_TOP + legend_height
        _text(canvas, "OBJECTS / TILE COORDINATES", (sx, row_top), 20)
        _text(canvas, "ID", (sx + 30, row_top + 27), 16, MUTED)
        _text(canvas, "(x,y)   w x h", (sx + 205, row_top + 27), 16, MUTED)
        for i, obj in enumerate(world.objects, 1):
            y = row_top + 52 + (i - 1) * 25
            pygame.draw.line(canvas, (218, 225, 222), (sx, y + 21), (sx + SIDEBAR - 16, y + 21))
            _text(canvas, str(i), (sx, y), 17, MUTED)
            _text(canvas, obj.id, (sx + 30, y), 17, max_width=165)
            x, yy = obj.position
            sw, sh = obj.attributes.size
            _text(canvas, f"({x},{yy})   {sw} x {sh}", (sx + 205, y), 17, max_width=120)
        _text(
            canvas,
            "(0,0) top-left / X right / Y down",
            (MARGIN, MAP_TOP + map_height + 22),
            17,
            MUTED,
            max_width=map_width,
        )
        target = Path(destination)
        if target.suffix.lower() != ".png":
            raise ValueError("PNG output path must end in .png")
        target.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(canvas, str(target))
        return target
