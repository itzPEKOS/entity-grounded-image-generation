"""Agent 1: Breaks a text-to-image query into discrete entities and scene composition."""

from __future__ import annotations

import re
from typing import Optional

from src.models import EntityCategory, EntityExtractionResult, ExtractedEntity


class EntityExtractorAgent:
    """Decomposes a user prompt into structured entities requiring visual grounding."""

    def __init__(self, llm_client: Optional[object] = None, model: str = "gemini-3.8-flash"):
        self.llm_client = llm_client
        self.model = model

    def run(self, prompt: str) -> EntityExtractionResult:
        if self.llm_client is not None:
            return self._extract_with_llm(prompt)
        return self._extract_heuristic(prompt)

    def _extract_with_llm(self, prompt: str) -> EntityExtractionResult:
        from google.genai import types
        from src.gemini_helper import generate_with_fallback

        system_prompt = (
            "You are an Entity Extraction Agent for a grounded text-to-image pipeline. "
            "Break the user's image prompt into discrete entities (landmarks, characters, "
            "animals, artifacts, vehicles, logos, flora) and describe the overall scene composition."
        )
        response = generate_with_fallback(
            client=self.llm_client,
            primary_model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=EntityExtractionResult,
            ),
        )
        return EntityExtractionResult.model_validate_json(response.text)

    def _extract_heuristic(self, prompt: str) -> EntityExtractionResult:
        proper_spans = re.findall(
            r"\b(?:[A-Z][a-zA-Z0-9'-]+(?:\s+(?:of|the|and|de|la|van)?\s*[A-Z][a-zA-Z0-9'-]+)*)",
            prompt,
        )
        seen = set()
        entities = []
        for span in proper_spans:
            clean = span.strip()
            slug = re.sub(r"[^a-z0-9]+", "_", clean.lower()).strip("_")
            if not slug or slug in seen:
                continue
            seen.add(slug)
            entities.append(
                ExtractedEntity(
                    entity_id=slug,
                    name=clean,
                    category=self._guess_category(clean),
                    user_requested_context=f"Depicted as requested in: '{prompt}'",
                    requires_factual_lookup=True,
                )
            )

        if not entities:
            entities.append(
                ExtractedEntity(
                    entity_id="primary_subject",
                    name=prompt[:60],
                    category=EntityCategory.OTHER,
                    user_requested_context=prompt,
                    requires_factual_lookup=False,
                )
            )

        return EntityExtractionResult(
            original_prompt=prompt,
            scene_composition=f"Cohesive composition faithfully depicting: {prompt}",
            entities=entities,
        )

    @staticmethod
    def _guess_category(name: str) -> EntityCategory:
        lower = name.lower()
        if any(w in lower for w in ("temple", "wat", "tower", "bridge", "palace", "castle", "statue", "mount")):
            return EntityCategory.LANDMARK
        if any(w in lower for w in ("dog", "cat", "inu", "eagle", "tiger", "panda", "fox")):
            return EntityCategory.ANIMAL
        if any(w in lower for w in ("car", "train", "ship", "rocket", "aircraft", "tuk")):
            return EntityCategory.VEHICLE
        return EntityCategory.OTHER
