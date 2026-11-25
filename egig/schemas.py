"""Messages exchanged between the agents.

Every agent reads and writes these Pydantic models, so the whole pipeline state can be
saved as JSON after each step. Models whose names start with `Parsed`, plus
`VisualProfile` and `VlmVerdicts`, are also sent to Gemini as response schemas, so they
deliberately avoid default values, dicts and unions.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    landmark = "landmark"
    animal = "animal"
    person = "person"
    vehicle = "vehicle"
    object = "object"
    plant = "plant"
    other = "other"


# --------------------------------------------------------------------------------------
# Agent 1 (Query Parser) output. Also used as the Gemini response schema.
# --------------------------------------------------------------------------------------


class ParsedEntity(BaseModel):
    id: str = Field(description="snake_case identifier, e.g. 'shiba_inu'")
    name: str = Field(description="Canonical name, e.g. 'Shiba Inu'")
    category: Category
    search_query: str = Field(
        description="Best Wikipedia search query for this entity, usually its canonical name"
    )
    detection_label: str = Field(
        description="Generic lowercase noun an object detector understands, e.g. 'dog', 'temple'"
    )
    count: int = Field(description="Number of instances the user asked for, usually 1")
    attributes: List[str] = Field(
        description="Only attributes the user explicitly asked for (pose, action, clothing, colour)"
    )
    box: List[float] = Field(
        description="Planned layout box [x0, y0, x1, y1], normalized 0-1, origin top-left"
    )
    layer: int = Field(description="Depth layer: 0 background, 1 middle, 2 foreground")
    needs_reference: bool = Field(
        description="True for specific real-world entities whose appearance must be looked up"
    )
    rigid: bool = Field(
        description="True when the shape is fixed (landmarks, buildings, vehicles); false for animals, people"
    )


class Relation(BaseModel):
    subject: str = Field(description="Entity id")
    predicate: str = Field(description="Spatial or interaction relation, e.g. 'in front of'")
    object: str = Field(description="Entity id")


class SceneSpec(BaseModel):
    setting: str = Field(description="Where the scene takes place, e.g. 'riverside in Bangkok'")
    style: str = Field(description="Visual style; 'photorealistic' if the user did not say")
    requirements: List[str] = Field(
        description="Global requirements the user explicitly stated, e.g. 'at sunset', 'raining'"
    )


class ParsedQuery(BaseModel):
    english_prompt: str = Field(
        description="The request rewritten as one concise English image prompt (same meaning)"
    )
    scene: SceneSpec
    entities: List[ParsedEntity]
    relations: List[Relation]


# --------------------------------------------------------------------------------------
# Agent 2 (Entity Retriever) output
# --------------------------------------------------------------------------------------


class VisualProfile(BaseModel):
    """What an entity really looks like. Also used as the Gemini response schema."""

    summary: str = Field(description="One or two sentences on what the entity looks like")
    key_features: List[str] = Field(
        description="3-5 short, visually checkable features (shape, colour, material, markings)"
    )
    negative_traits: List[str] = Field(
        description="0-3 short noun phrases describing common WRONG depictions, e.g. 'floppy ears'"
    )
    confusables: List[str] = Field(description="0-4 entities that look similar and are often confused")
    prompt_keywords: str = Field(
        description="Compact comma-separated phrase (max 15 words) to describe it to an image model"
    )


class ReferenceImage(BaseModel):
    path: str
    crop_path: str
    url: str
    source: str  # "wikipedia-lead" or "commons"
    det_score: Optional[float] = None
    clip_score: Optional[float] = None
    crop_box: Optional[List[float]] = None


class EntitySpec(ParsedEntity):
    profile: Optional[VisualProfile] = None
    references: List[ReferenceImage] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class ConstraintSpec(BaseModel):
    """The central constraint spec: built by Agent 1, grounded by Agent 2."""

    prompt: str  # exactly what the user typed (any language)
    prompt_en: str = ""  # English version used for the image model (CLIP only reads English)
    scene: SceneSpec
    entities: List[EntitySpec] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)

    @property
    def image_prompt(self) -> str:
        return self.prompt_en or self.prompt

    def entity(self, entity_id: Optional[str]) -> Optional[EntitySpec]:
        for e in self.entities:
            if e.id == entity_id:
                return e
        return None


# --------------------------------------------------------------------------------------
# Agent 4 (Verifier)
# --------------------------------------------------------------------------------------


class CheckKind(str, Enum):
    presence = "presence"
    count = "count"
    position = "position"
    identity = "identity"
    attribute = "attribute"
    feature = "feature"
    scene = "scene"
    relation = "relation"
    integrity = "integrity"  # the image as a whole: one continuous scene, not a framed picture / collage


class Severity(str, Enum):
    critical = "critical"  # must pass
    major = "major"  # a share of these must pass (see Settings.major_pass_rate)


class Method(str, Enum):
    detector = "detector"  # Grounding DINO
    vlm = "vlm"  # Gemini looking at the image


class Status(str, Enum):
    passed = "PASS"
    failed = "FAIL"
    skipped = "SKIP"


class Check(BaseModel):
    id: str
    entity_id: Optional[str] = None
    kind: CheckKind
    severity: Severity
    method: Method
    description: str


class CheckResult(BaseModel):
    check_id: str
    entity_id: Optional[str] = None
    kind: CheckKind
    severity: Severity
    method: Method
    description: str
    status: Status
    confidence: Optional[float] = None
    evidence: str = ""
    fix_hint: str = ""

    @classmethod
    def of(cls, check: Check, status: Status, evidence: str, confidence: Optional[float] = None,
           fix_hint: str = "") -> "CheckResult":
        return cls(
            check_id=check.id,
            entity_id=check.entity_id,
            kind=check.kind,
            severity=check.severity,
            method=check.method,
            description=check.description,
            status=status,
            confidence=confidence,
            evidence=evidence,
            fix_hint=fix_hint,
        )


class Detection(BaseModel):
    entity_id: Optional[str] = None
    label: str
    score: float
    box: List[float]  # normalized [x0, y0, x1, y1]
    low_confidence: bool = False  # only found with the lower threshold (box_threshold_low)


class Verdict(BaseModel):
    passed: bool
    critical_passed: int
    critical_total: int
    major_passed: int
    major_total: int
    major_rate: float
    skipped: int
    explanation: str


class AttemptReport(BaseModel):
    attempt: int
    image_path: str
    results: List[CheckResult]
    detections: List[Detection] = Field(default_factory=list)
    ref_similarity: Dict[str, float] = Field(default_factory=dict)
    verdict: Verdict
    repair_notes: List[str] = Field(default_factory=list)
    vlm_error: str = ""  # set when Gemini could not be reached, so the VLM checks were not verified
    vlm_counts: Dict[str, int] = Field(default_factory=dict)  # Gemini's own count per entity (second opinion)

    def failed(self) -> List[CheckResult]:
        return [r for r in self.results if r.status == Status.failed]


class VlmVerdict(BaseModel):
    check_id: str
    passed: bool
    confidence: float = Field(description="0 to 1")
    evidence: str = Field(description="One short sentence describing what is visible")
    fix_hint: str = Field(description="If failed: short instruction for the image generator; else empty")
    visible: bool = Field(default=True, description="false when the part this check is about is outside the "
                                                    "image or hidden, so it cannot be judged")


class VlmCount(BaseModel):
    entity_id: str
    count: int = Field(description="How many of this entity are visible in IMAGE A, including partly hidden ones")


class VlmVerdicts(BaseModel):
    verdicts: List[VlmVerdict]
    counts: List[VlmCount] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Agent 3 (Image Generator)
# --------------------------------------------------------------------------------------


class GenParams(BaseModel):
    """Knobs Agent 3 turns between attempts. The repair policy edits these."""

    seed: int
    ip_scale: Dict[str, float] = Field(default_factory=dict)  # per entity: how strongly to copy the reference
    control_scale: float = 0.5  # how strongly to follow the reference edges (rigid entities only)
    emphasis: Dict[str, int] = Field(default_factory=dict)  # per entity: move it earlier in the prompt
    extra_prompt: List[str] = Field(default_factory=list)
    extra_negative: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)  # why these values changed (repair log)


class GenerationRecord(BaseModel):
    attempt: int
    backend: str
    image_path: str
    prompt: str
    negative_prompt: str
    params: GenParams
    control_path: Optional[str] = None
    layout_path: Optional[str] = None
