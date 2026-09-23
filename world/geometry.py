"""Integer tile geometry; right and bottom edges are exclusive."""

from world.schema import ObjectToken

GROUND_TYPES = frozenset({"river", "road", "dune", "courtyard"})
ENTITY_TYPES = frozenset({"monster", "npc", "vehicle"})


def layer(obj: ObjectToken) -> int:
    if obj.object_type in GROUND_TYPES:
        return 0
    return 2 if obj.object_type in ENTITY_TYPES else 1


def bounds(obj: ObjectToken) -> tuple[int, int, int, int]:
    x, y = obj.position
    width, height = obj.attributes.size
    return x, y, x + width, y + height


def overlaps(a: ObjectToken, b: ObjectToken) -> bool:
    ax, ay, ar, ab = bounds(a)
    bx, by, br, bb = bounds(b)
    return ax < br and bx < ar and ay < bb and by < ab


def contains(outer: ObjectToken, inner: ObjectToken) -> bool:
    ox, oy, right, bottom = bounds(outer)
    ix, iy, ir, ib = bounds(inner)
    return ox <= ix and oy <= iy and ir <= right and ib <= bottom


def tile_distance(a: ObjectToken, b: ObjectToken) -> int:
    """Minimum Chebyshev distance between occupied tiles; adjacency is 1."""
    ax, ay, ar, ab = bounds(a)
    bx, by, br, bb = bounds(b)
    return max(ax - (br - 1), bx - (ar - 1), ay - (bb - 1), by - (ab - 1), 0)


def shares_edge(a: ObjectToken, b: ObjectToken) -> bool:
    ax, ay, ar, ab = bounds(a)
    bx, by, br, bb = bounds(b)
    return ((ar == bx or br == ax) and max(ay, by) < min(ab, bb)) or (
        (ab == by or bb == ay) and max(ax, bx) < min(ar, br)
    )
