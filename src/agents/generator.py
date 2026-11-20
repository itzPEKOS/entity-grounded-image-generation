"""Agent 3: Compiles the ConstraintSpec into a grounded prompt and generates the image."""

from __future__ import annotations

import io
import json
import os
import random
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw
from src.models import GlobalConstraintSpec


class ImageGeneratorAgent:
    """Synthesizes entity constraints + remediation hints into a grounded image generation call."""

    def __init__(
        self,
        image_client: Optional[object] = None,
        model: str = "flux",
    ):
        self.image_client = image_client
        self.model = model

    def build_grounded_prompt(
        self,
        spec: GlobalConstraintSpec,
        remediation_hints: Optional[List[str]] = None,
    ) -> str:
        sections = [
            f"Primary Scene Request: {spec.original_prompt}",
            f"Composition & Lighting: {spec.scene_composition}",
            "Grounded Entity Visual Specifications:",
        ]
        for ent in spec.entity_specs:
            features_str = "; ".join(ent.key_visual_features)
            neg_str = "; ".join(ent.negative_constraints)
            sections.append(
                f"- {ent.name}: {ent.visual_summary} Must include: [{features_str}]. "
                + (f"Strictly avoid: [{neg_str}]." if neg_str else "")
            )

        if remediation_hints:
            sections.append("Critical Corrections From Prior Attempt:")
            for hint in remediation_hints:
                sections.append(f"- FIX: {hint}")

        return "\n".join(sections)

    def run(
        self,
        spec: GlobalConstraintSpec,
        output_path: Path,
        remediation_hints: Optional[List[str]] = None,
    ) -> tuple[Path, str]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grounded_prompt = self.build_grounded_prompt(spec, remediation_hints)

        # 1. Paid Gemini Native Image Generation (when IMAGE_MODEL starts with 'gemini')
        if self.image_client is not None and self.model.startswith("gemini"):
            from google.genai import types

            response = self.image_client.models.generate_content(
                model=self.model,
                contents=grounded_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="1:1"),
                ),
            )
            for part in response.parts:
                if part.inline_data is not None:
                    self._save_as_png(part.inline_data.data, output_path)
                    return output_path, grounded_prompt
            raise RuntimeError("Gemini did not return an image part in the response.")

        # 2. Live Mode Multi-Provider Cascade (Pollinations FLUX -> AI Horde)
        if self.image_client is not None:
            compact_prompt = self._build_compact_visual_prompt(spec, remediation_hints)

            try:
                seed = random.randint(1, 999999)
                encoded = quote(compact_prompt)
                url = (
                    f"https://image.pollinations.ai/prompt/{encoded}"
                    f"?width=1024&height=1024&seed={seed}&model={self.model}&nologo=true"
                )
                req = Request(url, headers={"User-Agent": "EntityGroundedImageGen/0.2"})
                with urlopen(req, timeout=45.0) as resp:
                    raw = resp.read()
                    if resp.status == 200 and len(raw) > 5000:
                        self._save_as_png(raw, output_path)
                        print("[Generator] Image created via Pollinations.ai.")
                        return output_path, grounded_prompt
            except Exception as err:
                print(f"[Generator] Pollinations skipped ({str(err)[:70]}), switching to AI Horde...")

            self._generate_via_ai_horde(compact_prompt, output_path)
            return output_path, grounded_prompt

        # 3. Offline Demo Fallback
        self._render_demo_canvas(spec, grounded_prompt, output_path)
        return output_path, grounded_prompt

    @classmethod
    def _generate_via_ai_horde(cls, prompt: str, output_path: Path) -> None:
        api_key = os.getenv("AI_HORDE_API_KEY", "0000000000")
        headers = {
            "apikey": api_key,
            "Content-Type": "application/json",
            "Client-Agent": "EntityGroundedImageGen:0.2:github",
        }
        payload = json.dumps(
            {
                "prompt": f"{prompt}, highly detailed, sharp focus, photorealistic",
                "params": {
                    "width": 768,
                    "height": 768,
                    "steps": 25,
                    "n": 1,
                },
                "nsfw": False,
                "censor_nsfw": True,
                "r2": True,
            }
        ).encode("utf-8")

        req = Request(
            "https://aihorde.net/api/v2/generate/async",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urlopen(req, timeout=30.0) as resp:
            job_data = json.loads(resp.read().decode("utf-8"))

        job_id = job_data["id"]
        print(f"[Generator] Queued on AI Horde (job {job_id[:8]}...), waiting for render...")

        for _ in range(45):
            time.sleep(4)
            check_req = Request(
                f"https://aihorde.net/api/v2/generate/check/{job_id}",
                headers=headers,
            )
            with urlopen(check_req, timeout=20.0) as resp:
                status = json.loads(resp.read().decode("utf-8"))

            if status.get("faulted"):
                raise RuntimeError("AI Horde generation job faulted.")
            if status.get("done"):
                break
        else:
            raise TimeoutError("AI Horde generation timed out.")

        status_req = Request(
            f"https://aihorde.net/api/v2/generate/status/{job_id}",
            headers=headers,
        )
        with urlopen(status_req, timeout=20.0) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        img_url = result["generations"][0]["img"]
        with urlopen(Request(img_url, headers={"User-Agent": "EntityGroundedImageGen/0.2"}), timeout=45.0) as img_resp:
            cls._save_as_png(img_resp.read(), output_path)
        print("[Generator] Image created via AI Horde.")

    @staticmethod
    def _save_as_png(raw_bytes: bytes, output_path: Path) -> None:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img.save(output_path, format="PNG")

    @staticmethod
    def _build_compact_visual_prompt(
        spec: GlobalConstraintSpec,
        remediation_hints: Optional[List[str]] = None,
    ) -> str:
        entity_traits = []
        for ent in spec.entity_specs:
            entity_traits.append(f"{ent.name}: {', '.join(ent.key_visual_features[:2])}")
        core = (
            f"Candid DSLR photograph, sharp focus f/8, {spec.original_prompt}, "
            f"{'; '.join(entity_traits)}"
        )
        if remediation_hints:
            short_fixes = "; ".join(h.replace("FIX:", "").strip() for h in remediation_hints[:3])
            return f"{core}, {short_fixes}"[:480]
        return core[:480]

    @staticmethod
    def _render_demo_canvas(
        spec: GlobalConstraintSpec, grounded_prompt: str, output_path: Path
    ) -> None:
        img = Image.new("RGB", (1024, 768), color=(24, 30, 44))
        draw = ImageDraw.Draw(img)
        draw.rectangle([24, 24, 1000, 744], outline=(99, 179, 237), width=3)
        draw.text((48, 48), "ENTITY-GROUNDED IMAGE GENERATION (DEMO RENDER)", fill=(144, 205, 244))
        draw.text((48, 90), f"Prompt: {spec.original_prompt[:95]}", fill=(237, 242, 247))

        y = 145
        for ent in spec.entity_specs:
            draw.text((48, y), f"[Entity] {ent.name}", fill=(104, 211, 145))
            y += 28
            for feat in ent.key_visual_features[:3]:
                draw.text((72, y), f"• {feat[:95]}", fill=(203, 213, 224))
                y += 24
            y += 14

        img.save(output_path, format="PNG")
