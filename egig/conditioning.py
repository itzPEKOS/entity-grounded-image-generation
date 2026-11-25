"""Turn the planned layout + reference photos into conditioning images for Agent 3.

- IP-Adapter masks: one black/white mask per entity. White = where that entity's reference
  photo is allowed to influence the image. Entities in front cut their area out of the
  entities behind them.
- ControlNet image: Canny edges of each *rigid* entity's reference (landmarks, buildings...)
  pasted into its planned box, so the generated structure follows the real one.
  Nothing in it may have a straight rectangular outline: SDXL follows edges literally, and a
  rectangle (where a reference photo was cut, or where a foreground box was cleared) comes back
  as a picture frame, a poster or a panel in the image. So reference edges fade out towards the
  sides of their photo, and a foreground entity clears a soft oval instead of its whole box.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .geometry import area, center, intersection, to_pixels
from .schemas import EntitySpec


def _back_to_front(entities: Iterable[EntitySpec]):
    # stable sort: same layer keeps the order the parser gave
    return sorted(entities, key=lambda e: e.layer)


def layer_masks(entities: Iterable[EntitySpec], width: int, height: int) -> Dict[str, Image.Image]:
    entities = list(entities)
    masks: Dict[str, Image.Image] = {}
    for e in entities:
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(to_pixels(e.box, width, height), fill=255)
        for other in entities:
            if other is e or area(intersection(e.box, other.box)) <= 0:
                continue
            if other.layer > e.layer:  # in front: hides this entity
                draw.rectangle(to_pixels(other.box, width, height), fill=0)
            elif other.layer == e.layer:  # side by side: split the overlap, each keeps its own half
                half = _overlap_half_of_other(e.box, other.box)
                if half is not None:
                    draw.rectangle(to_pixels(half, width, height), fill=0)
        if mask.getbbox() is None:  # completely hidden: fall back to the full box
            ImageDraw.Draw(mask).rectangle(to_pixels(e.box, width, height), fill=255)
        masks[e.id] = mask
    return masks


def _overlap_half_of_other(box, other) -> Optional[Tuple[float, float, float, float]]:
    """Two entities on the same layer whose boxes overlap (cat 0.15-0.42 next to dog 0.38-0.75):
    without this, both reference photos act on the overlap and their fur mixes. Cut the overlap
    down its middle, across its narrow side; return the half that belongs to `other`."""
    ix0, iy0, ix1, iy1 = intersection(box, other)
    (cx, cy), (ox, oy) = center(box), center(other)
    if (ix1 - ix0) <= (iy1 - iy0) and cx != ox:  # side by side: split left / right
        mid = (ix0 + ix1) / 2
        return (mid, iy0, ix1, iy1) if cx < ox else (ix0, iy0, mid, iy1)
    if cy != oy:  # one above the other: split top / bottom
        mid = (iy0 + iy1) / 2
        return (ix0, mid, ix1, iy1) if cy < oy else (ix0, iy0, ix1, mid)
    return None  # same centre: no side to give away


def fit_into_box(img: Image.Image, box_px: Tuple[int, int, int, int], align: str = "bottom") -> Tuple[Image.Image, Tuple[int, int]]:
    """Resize keeping the aspect ratio so it fits in the box; centre horizontally and
    sit on the bottom edge (things stand on the ground) or centre vertically."""
    x0, y0, x1, y1 = box_px
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    scale = min(bw / img.width, bh / img.height)
    w, h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    resized = img.resize((w, h), Image.LANCZOS)
    ox = x0 + (bw - w) // 2
    oy = y1 - h if align == "bottom" else y0 + (bh - h) // 2
    return resized, (ox, oy)


def canny(img: Image.Image, low: int = 100, high: int = 200) -> np.ndarray:
    import cv2

    gray = np.array(img.convert("L"))
    return cv2.Canny(gray, low, high)


def edge_fade(h: int, w: int, margin: float = 0.15) -> np.ndarray:
    """1 in the middle, falling smoothly to 0 at the four sides (over `margin` of the size).
    Multiplying the edges of a cut-out photo by it removes the straight cut lines."""
    def ramp(n: int) -> np.ndarray:
        d = np.minimum(np.arange(n), np.arange(n)[::-1]) + 0.5  # distance to the nearer side
        t = np.clip(d / max(1.0, margin * n), 0.0, 1.0)
        return t * t * (3 - 2 * t)  # smoothstep
    return np.outer(ramp(h), ramp(w))


def soft_oval(h: int, w: int, softness: float = 0.25) -> np.ndarray:
    """1 inside an oval that fills the box, 0 outside, with a soft rim: the area a foreground
    entity takes away from the edges behind it, without a rectangular hole."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx + 0.5) / w * 2 - 1) ** 2 + ((yy + 0.5) / h * 2 - 1) ** 2)  # 0 centre, 1 at the oval
    return np.clip((1.0 - r) / softness, 0.0, 1.0)


def control_image(entities: Iterable[EntitySpec], crops: Dict[str, Image.Image], width: int, height: int) -> Optional[Image.Image]:
    """Black canvas with white edges. Walk entities back to front: each one first clears a soft
    oval in its box (it hides what is behind it), then rigid ones draw their reference edges,
    faded towards the sides of the photo.
    Returns None when no rigid entity has a reference (ControlNet is then switched off)."""
    entities = _back_to_front(entities)
    if not any(e.rigid and e.id in crops for e in entities):
        return None
    canvas = np.zeros((height, width), dtype=np.float32)
    for e in entities:
        x0, y0, x1, y1 = to_pixels(e.box, width, height)
        if y1 > y0 and x1 > x0:
            canvas[y0:y1, x0:x1] *= 1.0 - soft_oval(y1 - y0, x1 - x0)
        if e.rigid and e.id in crops:
            fitted, (ox, oy) = fit_into_box(crops[e.id], (x0, y0, x1, y1), align="bottom")
            edges = canny(fitted).astype(np.float32) * edge_fade(fitted.height, fitted.width)
            region = canvas[oy:oy + edges.shape[0], ox:ox + edges.shape[1]]
            region[:] = np.maximum(region, edges[: region.shape[0], : region.shape[1]])
    out = np.clip(canvas, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([out] * 3, axis=-1))


def draw_layout(entities: Iterable[EntitySpec], width: int, height: int,
                background: Optional[Image.Image] = None) -> Image.Image:
    """Picture of the planned boxes (for the report)."""
    img = (background.convert("RGB").resize((width, height)) if background else
           Image.new("RGB", (width, height), (245, 245, 245)))
    draw = ImageDraw.Draw(img)
    palette = [(230, 57, 70), (29, 120, 220), (42, 157, 80), (240, 160, 0), (130, 60, 200)]
    for i, e in enumerate(_back_to_front(entities)):
        color = palette[i % len(palette)]
        draw.rectangle(to_pixels(e.box, width, height), outline=color, width=4)
        x0, y0, _, _ = to_pixels(e.box, width, height)
        draw.text((x0 + 6, y0 + 4), f"{e.id} (layer {e.layer}{', rigid' if e.rigid else ''})", fill=color)
    return img
