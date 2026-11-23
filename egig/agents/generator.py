"""Agent 3 - Image Generator: constraint spec + current knobs -> image.

Builds the SDXL prompt (within CLIP's 77-token limit), the IP-Adapter masks and the
ControlNet edge image, then calls the backend. On FAIL, `repair()` turns the verifier's
failed checks into new knobs for the next attempt (see repair.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

from PIL import Image

from ..backends import Backend, GenerationJob, LayoutItem
from ..conditioning import control_image, draw_layout, layer_masks
from ..config import Settings
from ..repair import repair
from ..schemas import AttemptReport, ConstraintSpec, GenerationRecord, GenParams

TOKEN_BUDGET = 75  # CLIP reads 77 tokens including the start and end tokens
BASE_NEGATIVE = ["blurry", "low quality", "deformed", "distorted", "watermark", "text", "extra limbs"]
QUALITY = "highly detailed, sharp focus"


def _fit(segments: List[str], count_tokens: Callable[[str], int], budget: int = TOKEN_BUDGET) -> str:
    """Join segments in priority order, skipping any that would overflow the budget. The
    first segment (the user's own words) is always kept, truncated if it alone is too long."""
    out: List[str] = []
    for i, seg in enumerate(s.strip() for s in segments):
        if not seg or seg in out:
            continue
        candidate = ", ".join(out + [seg])
        if count_tokens(candidate) <= budget:
            out.append(seg)
        elif i == 0:
            words = seg.split()
            while words and count_tokens(" ".join(words)) > budget:
                words.pop()
            out.append(" ".join(words))
    return ", ".join(out)


def build_prompt(spec: ConstraintSpec, params: GenParams, count_tokens: Callable[[str], int]) -> str:
    segments = [spec.image_prompt]
    segments += params.extra_prompt  # corrections from the feedback loop, newest first
    ranked = sorted(spec.entities, key=lambda e: (-params.emphasis.get(e.id, 0), -e.layer))
    for e in ranked:
        if params.emphasis.get(e.id, 0) > 0 and e.attributes:
            segments.append(f"{e.name} {' '.join(e.attributes)}")
        if e.profile and e.profile.prompt_keywords:
            segments.append(e.profile.prompt_keywords)
    if spec.scene.style and spec.scene.style.lower() not in spec.image_prompt.lower():
        segments.append(spec.scene.style)
    segments.append(QUALITY)
    return _fit(segments, count_tokens)


def build_negative(spec: ConstraintSpec, params: GenParams, count_tokens: Callable[[str], int]) -> str:
    requested = " ".join(a.lower() for e in spec.entities for a in e.attributes) + " " + spec.image_prompt.lower()
    segments = list(params.extra_negative)
    for e in spec.entities:
        for neg in (e.profile.negative_traits if e.profile else []):
            # never forbid something the user asked for (e.g. user wants "floppy ears")
            if not any(w in requested for w in neg.lower().split() if len(w) > 3):
                segments.append(neg)
    segments += BASE_NEGATIVE
    if "photo" in spec.scene.style.lower():
        segments.append("cartoon, illustration")
    return _fit(segments, count_tokens)


class ImageGeneratorAgent:
    name = "ImageGenerator"

    def __init__(self, backend: Backend, settings: Settings):
        self.backend = backend
        self.s = settings

    def initial_params(self, spec: ConstraintSpec) -> GenParams:
        return GenParams(
            seed=self.s.base_seed,
            ip_scale={e.id: self.s.ip_scale for e in spec.entities if e.references},
            control_scale=self.s.control_scale,
        )

    def run(self, spec: ConstraintSpec, params: GenParams, out_dir: Path, attempt: int) -> GenerationRecord:
        out_dir.mkdir(parents=True, exist_ok=True)
        size = self.s.image_size
        crops: Dict[str, Image.Image] = {
            e.id: Image.open(e.references[0].crop_path).convert("RGB") for e in spec.entities if e.references
        }
        masks = layer_masks(spec.entities, size, size)
        control = control_image(spec.entities, crops, size, size)
        prompt = build_prompt(spec, params, self.backend.count_tokens)
        negative = build_negative(spec, params, self.backend.count_tokens)
        ip_entities = [e for e in spec.entities if e.id in crops]

        job = GenerationJob(
            prompt=prompt,
            negative_prompt=negative,
            width=size,
            height=size,
            steps=self.s.steps,
            guidance=self.s.guidance_scale,
            seed=params.seed,
            control_image=control,
            control_scale=params.control_scale if control is not None else 0.0,
            control_end=self.s.control_end,
            ip_images=[crops[e.id] for e in ip_entities],
            ip_masks=[masks[e.id] for e in ip_entities],
            ip_scales=[params.ip_scale.get(e.id, self.s.ip_scale) for e in ip_entities],
            layout=[LayoutItem(e.id, e.name, e.box, e.layer, crops.get(e.id)) for e in spec.entities],
        )
        print(f"[generator] attempt {attempt}: backend={self.backend.name}, seed={params.seed}, "
              f"ip={dict(zip([e.id for e in ip_entities], job.ip_scales))}, control={job.control_scale:.2f}")
        image = self.backend.generate(job)

        image_path = out_dir / "image.png"
        image.save(image_path)
        (out_dir / "prompt.txt").write_text(f"PROMPT:\n{prompt}\n\nNEGATIVE:\n{negative}\n", encoding="utf-8")
        layout_path = out_dir / "layout.png"
        draw_layout(spec.entities, size, size).save(layout_path)
        control_path = None
        if control is not None:
            control_path = out_dir / "control_canny.png"
            control.save(control_path)
        return GenerationRecord(
            attempt=attempt,
            backend=self.backend.name,
            image_path=str(image_path),
            prompt=prompt,
            negative_prompt=negative,
            params=params,
            control_path=str(control_path) if control_path else None,
            layout_path=str(layout_path),
        )

    def repair(self, params: GenParams, report: AttemptReport, spec: ConstraintSpec) -> GenParams:
        return repair(params, report, spec)
