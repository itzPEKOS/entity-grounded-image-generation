"""Turn the planned layout + reference photos into conditioning images for Agent 3.

- IP-Adapter masks: one black/white mask per entity. White = where that entity's reference
  photo is allowed to influence the image. Entities in front cut their area out of the
  entities behind them.
- ControlNet image: Canny edges of each *rigid* entity's reference (landmarks, buildings...)
  pasted into its planned box, so the generated structure follows the real one.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .geometry import intersection, area, to_pixels
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
            if other is not e and other.layer > e.layer and area(intersection(e.box, other.box)) > 0:
                draw.rectangle(to_pixels(other.box, width, height), fill=0)
        if mask.getbbox() is None:  # completely hidden: fall back to the full box
            ImageDraw.Draw(mask).rectangle(to_pixels(e.box, width, height), fill=255)
        masks[e.id] = mask
    return masks


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


def control_image(entities: Iterable[EntitySpec], crops: Dict[str, Image.Image], width: int, height: int) -> Optional[Image.Image]:
    """Black canvas with white edges. Walk entities back to front: each one first clears its
    own box (it hides what is behind it), then rigid ones draw their reference edges.
    Returns None when no rigid entity has a reference (ControlNet is then switched off)."""
    entities = _back_to_front(entities)
    if not any(e.rigid and e.id in crops for e in entities):
        return None
    canvas = np.zeros((height, width), dtype=np.uint8)
    for e in entities:
        x0, y0, x1, y1 = to_pixels(e.box, width, height)
        canvas[y0:y1, x0:x1] = 0
        if e.rigid and e.id in crops:
            fitted, (ox, oy) = fit_into_box(crops[e.id], (x0, y0, x1, y1), align="bottom")
            edges = canny(fitted)
            h, w = edges.shape
            region = canvas[oy:oy + h, ox:ox + w]
            canvas[oy:oy + h, ox:ox + w] = np.maximum(region, edges[: region.shape[0], : region.shape[1]])
    return Image.fromarray(np.stack([canvas] * 3, axis=-1))


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
