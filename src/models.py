"""Core Pydantic data models for Entity-Grounded Image Generation."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class EntityCategory(str, Enum):
    LANDMARK = "landmark"
    PERSON_OR_CHARACTER = "person_or_character"
    ANIMAL = "animal"
    OBJECT_OR_ARTIFACT = "object_or_artifact"
    VEHICLE = "vehicle"
    BRAND_OR_LOGO = "brand_or_logo"
    FLORA = "flora"
    OTHER = "other"


class ExtractedEntity(BaseModel):
    """A single entity parsed from the user's text-to-image query."""

    entity_id: str = Field(..., description="Unique slug identifier, e.g., 'wat_arun'")
    name: str = Field(..., description="Canonical name of the entity")
    category: EntityCategory = Field(default=EntityCategory.OTHER)
    user_requested_context: str = Field(
        ...,
        description="How the user wants this entity posed, styled, or placed in the scene",
    )
    requires_factual_lookup: bool = Field(
        default=True,
        description="Whether real-world/canonical visual lookup is needed",
    )


class EntityExtractionResult(BaseModel):
    """Output of the Entity Extractor Agent."""

    original_prompt: str
    scene_composition: str = Field(
        ..., description="Overall layout, lighting, mood, and spatial relationships"
    )
    entities: List[ExtractedEntity] = Field(default_factory=list)


class VisualCheckItem(BaseModel):
    """A concrete, verifiable visual criterion for an entity or scene."""

    check_id: str
    entity_id: Optional[str] = Field(
        default=None, description="Target entity_id, or None for global composition checks"
    )
    criterion: str = Field(..., description="Specific visual attribute to verify")
    importance: str = Field(
        default="critical", description="critical | major | minor"
    )


class EntityConstraintSpec(BaseModel):
    """Grounded visual specification for a single entity."""

    entity_id: str
    name: str
    visual_summary: str = Field(
        ..., description="Concise factual description of what the entity actually looks like"
    )
    key_visual_features: List[str] = Field(
        default_factory=list,
        description="Mandatory colors, shapes, materials, structural details",
    )
    negative_constraints: List[str] = Field(
        default_factory=list,
        description="Common hallucinations or inaccurate traits to explicitly avoid",
    )
    reference_sources: List[str] = Field(
        default_factory=list,
        description="URLs or knowledge sources consulted during visual lookup",
    )
    checks: List[VisualCheckItem] = Field(default_factory=list)


class GlobalConstraintSpec(BaseModel):
    """Complete constraint specification guiding generation and verification."""

    original_prompt: str
    scene_composition: str
    entity_specs: List[EntityConstraintSpec] = Field(default_factory=list)
    global_checks: List[VisualCheckItem] = Field(default_factory=list)

    def all_checks(self) -> List[VisualCheckItem]:
        items: List[VisualCheckItem] = []
        for spec in self.entity_specs:
            items.extend(spec.checks)
        items.extend(self.global_checks)
        return items


class CheckVerdict(BaseModel):
    """Pass/fail evaluation for a single visual check."""

    check_id: str
    entity_id: Optional[str] = None
    criterion: str
    passed: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    observed_evidence: str = Field(
        ..., description="What the verifier actually sees in the generated image"
    )
    remediation_hint: Optional[str] = Field(
        default=None, description="How to adjust the prompt if passed=False"
    )


class VerificationReport(BaseModel):
    """Structured pass/fail audit report for a generated image."""

    attempt: int
    image_path: str
    overall_passed: bool
    pass_rate: float = Field(..., ge=0.0, le=1.0)
    verdicts: List[CheckVerdict] = Field(default_factory=list)
    summary: str

    def failed_checks(self) -> List[CheckVerdict]:
        return [v for v in self.verdicts if not v.passed]

    def to_markdown(self) -> str:
        status_badge = "PASS" if self.overall_passed else "FAIL"
        lines = [
            f"# Verification Report (Attempt {self.attempt}) — [{status_badge}]",
            "",
            f"- **Image:** `{self.image_path}`",
            f"- **Overall Status:** **{status_badge}**",
            f"- **Pass Rate:** {self.pass_rate * 100:.1f}% ({sum(v.passed for v in self.verdicts)}/{len(self.verdicts)} checks)",
            f"- **Summary:** {self.summary}",
            "",
            "## Check-by-Check Breakdown",
            "",
            "| Check ID | Entity | Criterion | Status | Confidence | Observed Evidence |",
            "| :--- | :--- | :--- | :---: | :---: | :--- |",
        ]
        for v in self.verdicts:
            badge = "PASS" if v.passed else "FAIL"
            ent = v.entity_id or "global"
            lines.append(
                f"| `{v.check_id}` | `{ent}` | {v.criterion} | **{badge}** | {v.confidence:.2f} | {v.observed_evidence} |"
            )
        failed = self.failed_checks()
        if failed:
            lines.extend(["", "## Remediation Hints", ""])
            for f in failed:
                lines.append(
                    f"- **`{f.check_id}` ({f.entity_id or 'global'}):** {f.remediation_hint or 'Emphasize this visual feature more strongly in the prompt.'}"
                )
        return "\n".join(lines) + "\n"
