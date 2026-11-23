"""Box helpers. Boxes are normalized [x0, y0, x1, y1] with the origin at the top-left."""

from __future__ import annotations

from typing import List, Sequence, Tuple, TypeVar

Box = List[float]


def clamp_box(box: Sequence[float], min_size: float = 0.05) -> Box:
    """Clamp to [0, 1], fix swapped corners and enforce a minimum width/height."""
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
    y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
    if x1 - x0 < min_size:
        cx = (x0 + x1) / 2
        x0, x1 = max(0.0, cx - min_size / 2), min(1.0, cx + min_size / 2)
    if y1 - y0 < min_size:
        cy = (y0 + y1) / 2
        y0, y1 = max(0.0, cy - min_size / 2), min(1.0, cy + min_size / 2)
    return [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


def area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Sequence[float], b: Sequence[float]) -> Box:
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = area(intersection(a, b))
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def center(b: Sequence[float]) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def expand(b: Sequence[float], margin: float) -> Box:
    """Grow a box by `margin` (fraction of the image) on every side, clamped to the image."""
    return [max(0.0, b[0] - margin), max(0.0, b[1] - margin), min(1.0, b[2] + margin), min(1.0, b[3] + margin)]


def contains_point(b: Sequence[float], p: Tuple[float, float], margin: float = 0.0) -> bool:
    g = expand(b, margin)
    return g[0] <= p[0] <= g[2] and g[1] <= p[1] <= g[3]


def union_box(boxes: Sequence[Sequence[float]]) -> Box:
    """Smallest box containing all boxes (e.g. every tower of a temple = the whole temple)."""
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def to_pixels(b: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    return (round(b[0] * width), round(b[1] * height), round(b[2] * width), round(b[3] * height))


def normalize_pixels(b: Sequence[float], width: int, height: int) -> Box:
    return clamp_box([b[0] / width, b[1] / height, b[2] / width, b[3] / height], min_size=0.0)


T = TypeVar("T")


def nms(items: List[T], iou_threshold: float = 0.5) -> List[T]:
    """Non-maximum suppression for objects with `.score` and `.box`: keep the best box of
    every cluster of boxes that overlap more than `iou_threshold`."""
    kept: List[T] = []
    for item in sorted(items, key=lambda d: d.score, reverse=True):  # type: ignore[attr-defined]
        if all(iou(item.box, k.box) <= iou_threshold for k in kept):  # type: ignore[attr-defined]
            kept.append(item)
    return kept
