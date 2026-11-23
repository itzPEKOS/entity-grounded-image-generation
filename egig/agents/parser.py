"""Agent 1 - Query Parser: user text -> constraint spec (entities, layout, scene, relations)."""

from __future__ import annotations

import re
from typing import Dict, List

from ..geometry import clamp_box
from ..llm import LLM
from ..schemas import ConstraintSpec, EntitySpec, ParsedQuery, Relation, SceneSpec

MAX_ENTITIES = 4

SYSTEM = """You are the Query Parser agent of an entity-grounded text-to-image system.
Turn the user's image request into a constraint spec (JSON). The request may be in any language
(e.g. Thai); write every field in English. english_prompt = the request as one concise English prompt.

Entities
- An entity is a concrete visible thing the user mentions: a landmark, animal, person, vehicle, object, plant.
- At most 4 entities. Do not make the setting itself an entity unless it is a specific named place (e.g. "Wat Arun").
- needs_reference = true for specific real-world things whose look must be looked up (a named landmark, a dog breed,
  a car model, a species). false for generic things ("a red ball", "a wooden table").
- search_query: the exact English Wikipedia article title when you know it, disambiguated if the name is
  ambiguous (e.g. "Mercury (planet)", not "Mercury"); otherwise the canonical name.
- rigid = true when the shape is fixed (landmarks, buildings, bridges, vehicles, statues); false for animals and people.
- detection_label: a short generic lowercase noun an object detector knows: "dog", "cat", "temple", "tower",
  "bridge", "car", "person", "bird". Never a proper name.
- attributes: ONLY what the user explicitly asked for about that entity (pose, action, clothing, colour, size).
  Do not invent attributes. Empty list if none.
- count: how many the user asked for (default 1).

Layout
- box = [x0, y0, x1, y1], normalized 0..1, origin at the top-left. Each box at least 0.2 wide and 0.2 tall.
- Background landmarks/buildings: large, upper-middle, layer 0, e.g. [0.10, 0.05, 0.90, 0.75].
- Foreground subjects (animals, people): lower part, in front, layer 2, e.g. [0.30, 0.45, 0.70, 0.98].
- Respect explicit positions ("on the left" -> x in 0..0.5). Avoid overlapping foreground boxes.

Scene
- setting: where it happens. style: "photorealistic" unless the user asked for another style.
- requirements: global conditions the user explicitly stated (time of day, weather, lighting, season, style).
  Each requirement short, e.g. "at sunset". Empty list if none.

Relations
- Spatial/interaction relations the user stated between entities, using entity ids, e.g.
  {"subject": "shiba_inu", "predicate": "sitting in front of", "object": "wat_arun"}.

Example - request: "A Shiba Inu sitting in front of Wat Arun at sunset"
{"english_prompt": "A Shiba Inu sitting in front of Wat Arun at sunset",
 "scene":{"setting": "riverside in Bangkok", "style": "photorealistic", "requirements": ["at sunset"]},
 "entities": [
  {"id": "wat_arun", "name": "Wat Arun", "category": "landmark", "search_query": "Wat Arun",
   "detection_label": "temple", "count": 1, "attributes": [], "box": [0.12, 0.03, 0.88, 0.72],
   "layer": 0, "needs_reference": true, "rigid": true},
  {"id": "shiba_inu", "name": "Shiba Inu", "category": "animal", "search_query": "Shiba Inu",
   "detection_label": "dog", "count": 1, "attributes": ["sitting"], "box": [0.32, 0.48, 0.68, 0.98],
   "layer": 2, "needs_reference": true, "rigid": false}],
 "relations": [{"subject": "shiba_inu", "predicate": "sitting in front of", "object": "wat_arun"}]}
"""


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "entity"


def default_box(index: int, total: int, layer: int) -> List[float]:
    if layer == 0:
        return [0.10, 0.05, 0.90, 0.75]
    width = 0.8 / max(1, total)
    x0 = 0.1 + index * width
    return [x0, 0.45, x0 + width, 0.95]


def normalize(prompt: str, parsed: ParsedQuery) -> ConstraintSpec:
    """Make the LLM output safe to use: unique ids, valid boxes, known relation ids..."""
    entities: List[EntitySpec] = []
    id_map: Dict[str, str] = {}
    raw_entities = parsed.entities[:MAX_ENTITIES]
    for i, pe in enumerate(raw_entities):
        base = slugify(pe.id or pe.name)
        eid, k = base, 2
        while any(e.id == eid for e in entities):
            eid, k = f"{base}_{k}", k + 1
        for key in (slugify(pe.id), slugify(pe.name)):
            id_map.setdefault(key, eid)  # first entity wins, so relations keep pointing at it

        layer = min(max(int(pe.layer), 0), 2)
        box = pe.box if len(pe.box) == 4 else default_box(i, len(raw_entities), layer)
        box = clamp_box(box, min_size=0.15)
        label = re.sub(r"[^a-z0-9 \-]", "", pe.detection_label.lower()).strip() or pe.category.value
        attributes = list(dict.fromkeys(a.strip() for a in pe.attributes if a.strip()))

        entities.append(EntitySpec(
            id=eid,
            name=pe.name.strip() or eid.replace("_", " "),
            category=pe.category,
            search_query=pe.search_query.strip() or pe.name.strip(),
            detection_label=" ".join(label.split()[:3]),
            count=min(max(int(pe.count), 1), 6),
            attributes=attributes,
            box=box,
            layer=layer,
            needs_reference=bool(pe.needs_reference),
            rigid=bool(pe.rigid),
        ))

    relations = []
    for r in parsed.relations:
        s, o = id_map.get(slugify(r.subject)), id_map.get(slugify(r.object))
        if s and o and s != o and r.predicate.strip():
            relations.append(Relation(subject=s, predicate=r.predicate.strip(), object=o))

    scene = SceneSpec(
        setting=parsed.scene.setting.strip(),
        style=parsed.scene.style.strip() or "photorealistic",
        requirements=list(dict.fromkeys(q.strip() for q in parsed.scene.requirements if q.strip())),
    )
    return ConstraintSpec(prompt=prompt.strip(), prompt_en=parsed.english_prompt.strip(), scene=scene,
                          entities=entities, relations=relations)


class QueryParserAgent:
    name = "QueryParser"

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, prompt: str) -> ConstraintSpec:
        parsed = self.llm.generate_json(ParsedQuery, [f"User request: {prompt}"], system=SYSTEM)
        return normalize(prompt, parsed)
