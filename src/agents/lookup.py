"""Agent 2: Looks up real-world visual appearance for each entity and builds ConstraintSpec."""

from __future__ import annotations

import json
from typing import Dict, List, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

from src.models import (
    EntityConstraintSpec,
    EntityExtractionResult,
    ExtractedEntity,
    GlobalConstraintSpec,
    VisualCheckItem,
)

BUILTIN_VISUAL_KB: Dict[str, Dict[str, object]] = {
    "wat_arun": {
        "visual_summary": (
            "Wat Arun Ratchawararam (Temple of Dawn) in Bangkok, featuring a tall central Khmer-style "
            "prang spire encrusted with colorful porcelain and seashells, flanked by four smaller prangs "
            "along the Chao Phraya River."
        ),
        "key_visual_features": [
            "Tall central Khmer-style prang tower rising sharply with tiered ornamentation",
            "Off-white / cream stucco surface inlaid with intricate floral mosaics of colorful Chinese porcelain",
            "Four smaller satellite prangs surrounding the central spire",
            "Steep staircases and guardian yaksha/kinnari sculptures along the terraces",
        ],
        "negative_constraints": [
            "Do not render as a golden Thai chedi or flat-roofed pagoda",
            "Avoid solid neon-red or plain gray concrete walls without porcelain mosaic texture",
        ],
        "reference_sources": ["https://en.wikipedia.org/wiki/Wat_Arun"],
    },
    "shiba_inu": {
        "visual_summary": (
            "Small-to-medium Japanese spitz-type dog with a compact muscular frame, prick triangular ears, "
            "fox-like face with cream urajiro markings, and a thick curled tail over the back."
        ),
        "key_visual_features": [
            "Upright triangular prick ears slightly tilted forward",
            "Cream-white urajiro markings on cheeks, muzzle, chest, and underside",
            "Thick double coat (red-sesame, black-and-tan, or cream) with a tightly curled sickle tail",
        ],
        "negative_constraints": [
            "Do not render with floppy ears or a long straight tail",
        ],
        "reference_sources": ["https://en.wikipedia.org/wiki/Shiba_Inu"],
    },
    "golden_gate_bridge": {
        "visual_summary": (
            "Art Deco suspension bridge spanning the Golden Gate strait in San Francisco, famous for its "
            "International Orange steel towers, vertical suspender ropes, and sweeping main cables."
        ),
        "key_visual_features": [
            "Distinctive International Orange (vermilion-orange) paint finish",
            "Two stepped Art Deco steel towers with horizontal cross-bracing",
            "Parabolic main suspension cables with vertical suspender ropes",
        ],
        "negative_constraints": [
            "Do not paint the bridge silver, blue, or bright yellow",
            "Do not render as a stone arch or cable-stayed bridge",
        ],
        "reference_sources": ["https://en.wikipedia.org/wiki/Golden_Gate_Bridge"],
    },
}


class VisualLookupAgent:
    """Queries external knowledge sources + Gemini Grounding to synthesize visual constraints."""

    def __init__(self, llm_client: Optional[object] = None, model: str = "gemini-3.8-flash"):
        self.llm_client = llm_client
        self.model = model

    def run(self, extraction: EntityExtractionResult) -> GlobalConstraintSpec:
        entity_specs: List[EntityConstraintSpec] = [
            self._lookup_entity(entity) for entity in extraction.entities
        ]
        global_checks = [
            VisualCheckItem(
                check_id="global_prompt_alignment",
                entity_id=None,
                criterion=f"Scene faithfully satisfies the user request: '{extraction.original_prompt}'",
                importance="critical",
            ),
            VisualCheckItem(
                check_id="global_coherence",
                entity_id=None,
                criterion="Lighting, perspective, and scale between all entities are visually coherent",
                importance="major",
            ),
        ]
        return GlobalConstraintSpec(
            original_prompt=extraction.original_prompt,
            scene_composition=extraction.scene_composition,
            entity_specs=entity_specs,
            global_checks=global_checks,
        )

    def _lookup_entity(self, entity: ExtractedEntity) -> EntityConstraintSpec:
        if entity.entity_id in BUILTIN_VISUAL_KB:
            kb = BUILTIN_VISUAL_KB[entity.entity_id]
            features = list(kb["key_visual_features"])
            checks = [
                VisualCheckItem(
                    check_id=f"{entity.entity_id}_check_{i+1}",
                    entity_id=entity.entity_id,
                    criterion=feat,
                    importance="critical" if i == 0 else "major",
                )
                for i, feat in enumerate(features)
            ]
            return EntityConstraintSpec(
                entity_id=entity.entity_id,
                name=entity.name,
                visual_summary=str(kb["visual_summary"]),
                key_visual_features=features,
                negative_constraints=list(kb["negative_constraints"]),
                reference_sources=list(kb["reference_sources"]),
                checks=checks,
            )

        wiki_summary, wiki_url = self._fetch_wikipedia_summary(entity.name)
        if self.llm_client is not None:
            return self._synthesize_with_llm(entity, wiki_summary, wiki_url)

        summary_text = wiki_summary or f"Canonical visual representation of {entity.name}."
        features = [
            f"Accurate recognizable silhouette and proportions of {entity.name}",
            f"Authentic surface materials, colors, and details matching {entity.name}",
        ]
        checks = [
            VisualCheckItem(
                check_id=f"{entity.entity_id}_check_{i+1}",
                entity_id=entity.entity_id,
                criterion=feat,
                importance="critical" if i == 0 else "major",
            )
            for i, feat in enumerate(features)
        ]
        return EntityConstraintSpec(
            entity_id=entity.entity_id,
            name=entity.name,
            visual_summary=summary_text,
            key_visual_features=features,
            negative_constraints=[f"Avoid generic or distorted approximations of {entity.name}"],
            reference_sources=[wiki_url] if wiki_url else [],
            checks=checks,
        )

    def _synthesize_with_llm(
        self, entity: ExtractedEntity, wiki_summary: str, wiki_url: str
    ) -> EntityConstraintSpec:
        from google.genai import types
        from google.genai.errors import APIError
        from src.gemini_helper import generate_with_fallback

        prompt = (
            f"Entity: {entity.name} (category: {entity.category.value})\n"
            f"User context: {entity.user_requested_context}\n"
            f"Reference knowledge: {wiki_summary or 'Use verified real-world visual knowledge.'}\n\n"
            "Produce a strict EntityConstraintSpec with concrete, observable visual features "
            "(colors, geometry, textures, markings), negative constraints, and 2-4 verifiable VisualCheckItems."
        )
        try:
            response = generate_with_fallback(
                client=self.llm_client,
                primary_model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a Visual Knowledge & Constraint Specification Agent.",
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    response_mime_type="application/json",
                    response_schema=EntityConstraintSpec,
                ),
            )
        except APIError:
            response = generate_with_fallback(
                client=self.llm_client,
                primary_model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a Visual Knowledge & Constraint Specification Agent.",
                    response_mime_type="application/json",
                    response_schema=EntityConstraintSpec,
                ),
            )
        spec = EntityConstraintSpec.model_validate_json(response.text)
        if wiki_url and wiki_url not in spec.reference_sources:
            spec.reference_sources.append(wiki_url)
        return spec

    @staticmethod
    def _fetch_wikipedia_summary(query: str) -> tuple[str, str]:
        slug = quote(query.strip().replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        headers = {"User-Agent": "EntityGroundedImageGen/0.2"}
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("extract", ""), data.get("content_urls", {}).get("desktop", {}).get("page", "")
        except Exception:
            pass
        return "", ""
