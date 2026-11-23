"""Laptop stand-in for the diffusion model (see backends/__init__.py)."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

from ..conditioning import fit_into_box
from ..geometry import to_pixels
from . import GenerationJob, estimate_tokens

SKIES = {
    "sunset": ((255, 140, 66), (120, 60, 130)),
    "sunrise": ((255, 190, 120), (110, 160, 220)),
    "night": ((10, 20, 60), (40, 40, 90)),
}
DEFAULT_SKY = ((150, 200, 245), (110, 160, 110))


class CollageBackend:
    name = "collage"

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def generate(self, job: GenerationJob) -> Image.Image:
        top, bottom = next((v for k, v in SKIES.items() if k in job.prompt.lower()), DEFAULT_SKY)
        img = Image.new("RGB", (job.width, job.height))
        draw = ImageDraw.Draw(img)
        for y in range(job.height):  # vertical gradient background
            t = y / max(1, job.height - 1)
            draw.line([(0, y), (job.width, y)], fill=tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)))

        rng = random.Random(job.seed)
        for item in sorted(job.layout, key=lambda i: i.layer):
            dx, dy = (rng.uniform(-0.02, 0.02) for _ in range(2))
            box = [item.box[0] + dx, item.box[1] + dy, item.box[2] + dx, item.box[3] + dy]
            px = to_pixels(box, job.width, job.height)
            if item.crop is not None:
                fitted, offset = fit_into_box(item.crop.convert("RGB"), px, align="bottom")
                img.paste(fitted, offset)
            else:
                draw.rectangle(px, outline=(40, 40, 40), width=3, fill=(220, 220, 220))
                draw.text((px[0] + 8, px[1] + 8), item.label, fill=(20, 20, 20))
        draw.rectangle([0, 0, job.width, 22], fill=(0, 0, 0))
        draw.text((8, 5), "COLLAGE STUB (no diffusion) - for testing the pipeline without a GPU", fill=(255, 255, 0))
        return img
