"""Settings. Every field can be overridden by an environment variable (or a line in .env)
with the same name in UPPERCASE, e.g. `max_attempts` <- MAX_ATTEMPTS=4."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any


@dataclass
class Settings:
    # --- Gemini (Agents 1, 2 and the VLM part of Agent 4) ---
    gemini_api_key: str = ""
    llm_model: str = "gemini-3.5-flash-lite"
    vlm_model: str = "gemini-3.5-flash-lite"
    llm_min_interval_s: float = 4.5  # 60 / 4.5 = 13.3 calls per minute, under the ~15 RPM free tier
    llm_max_retries: int = 5
    llm_temperature: float = 0.2
    # used when the main model is overloaded (503) or out of quota (429):
    # "auto" = stable gemini-X-flash(-lite) models your key can list, "none" = off, or "model-a,model-b"
    llm_fallback_models: str = "auto"

    # --- Pipeline ---
    output_dir: str = "outputs"
    max_attempts: int = 3  # first image + 2 repairs
    generator: str = "auto"  # auto | sdxl | collage
    detector: str = "auto"  # auto | gdino | none

    # --- Agent 2: reference retrieval ---
    refs_per_entity: int = 3
    commons_candidates: int = 8
    wiki_lang: str = "en"
    # Wikimedia blocks requests (HTTP 403) whose User-Agent has no contact info.
    # Put your email or a URL (e.g. your GitHub profile) here.
    wiki_contact: str = ""

    # --- Agent 3: generation ---
    image_size: int = 1024
    steps: int = 25
    guidance_scale: float = 6.0
    base_seed: int = 1234
    ip_scale: float = 0.6  # IP-Adapter strength per reference image
    control_scale: float = 0.5  # ControlNet strength
    control_end: float = 0.6  # ControlNet only guides the first 60% of denoising steps
    sdxl_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_model: str = "madebyollin/sdxl-vae-fp16-fix"
    controlnet_model: str = "diffusers/controlnet-canny-sdxl-1.0"
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "sdxl_models"
    ip_adapter_weight: str = "ip-adapter-plus_sdxl_vit-h.safetensors"
    ip_image_encoder_subfolder: str = "models/image_encoder"
    cpu_offload: bool = True  # needed to fit SDXL + ControlNet + IP-Adapter on a 16 GB T4

    # --- Agent 4: verification ---
    gdino_model: str = "IDEA-Research/grounding-dino-tiny"
    clip_model: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    ref_box_threshold: float = 0.30
    position_margin: float = 0.10
    major_pass_rate: float = 0.70

    @classmethod
    def from_env(cls, **overrides: Any) -> "Settings":
        values = {}
        for f in fields(cls):
            raw = os.getenv(f.name.upper())
            values[f.name] = f.default if raw is None or raw.strip() == "" else _cast(raw, f.default)
        settings = cls(**values)
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(settings, **clean) if clean else settings


def _cast(raw: str, default: Any) -> Any:
    raw = raw.strip()
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw
