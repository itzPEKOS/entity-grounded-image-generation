"""Image generation backends used by Agent 3.

- `SDXLBackend`  : the real one (SDXL + IP-Adapter + ControlNet). Needs a CUDA GPU.
- `CollageBackend`: a stand-in for laptops without a GPU. It pastes the reference photos
  into the planned boxes so the rest of the pipeline (verification, feedback loop) can be
  tested end to end. It is NOT diffusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from PIL import Image


@dataclass
class LayoutItem:
    entity_id: str
    label: str
    box: List[float]
    layer: int
    crop: Optional[Image.Image] = None


@dataclass
class GenerationJob:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    control_image: Optional[Image.Image] = None
    control_scale: float = 0.0
    control_end: float = 0.6
    ip_images: List[Image.Image] = field(default_factory=list)
    ip_masks: List[Image.Image] = field(default_factory=list)
    ip_scales: List[float] = field(default_factory=list)
    layout: List[LayoutItem] = field(default_factory=list)


class Backend(Protocol):
    name: str

    def count_tokens(self, text: str) -> int: ...

    def generate(self, job: GenerationJob) -> Image.Image: ...


def estimate_tokens(text: str) -> int:
    """Rough CLIP token count when the real tokenizer is not loaded: ~1.3 tokens per word,
    plus one per comma."""
    words = text.replace(",", " ").split()
    return round(len(words) * 1.3) + text.count(",")
