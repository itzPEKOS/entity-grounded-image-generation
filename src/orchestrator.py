"""Multi-agent orchestrator coordinating extraction, lookup, generation, and verification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.agents import (
    EntityExtractorAgent,
    ImageGeneratorAgent,
    VerificationAgent,
    VisualLookupAgent,
)
from src.models import (
    EntityExtractionResult,
    GlobalConstraintSpec,
    VerificationReport,
)


@dataclass
class PipelineRunResult:
    """Artifacts produced by a complete multi-agent run."""

    run_dir: Path
    extraction: EntityExtractionResult
    constraint_spec: GlobalConstraintSpec
    reports: List[VerificationReport]
    final_image_path: Path
    final_report: VerificationReport


class EntityGroundedOrchestrator:
    """Coordinates the 4-agent loop with iterative self-correction on verification failures."""

    def __init__(
        self,
        extractor: Optional[EntityExtractorAgent] = None,
        lookup: Optional[VisualLookupAgent] = None,
        generator: Optional[ImageGeneratorAgent] = None,
        verifier: Optional[VerificationAgent] = None,
        max_retries: int = 2,
        output_root: str | Path = "outputs",
    ):
        self.extractor = extractor or EntityExtractorAgent()
        self.lookup = lookup or VisualLookupAgent()
        self.generator = generator or ImageGeneratorAgent()
        self.verifier = verifier or VerificationAgent()
        self.max_retries = max_retries
        self.output_root = Path(output_root)

    def run(self, prompt: str, run_id: Optional[str] = None) -> PipelineRunResult:
        slug = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_root / slug
        run_dir.mkdir(parents=True, exist_ok=True)

        extraction = self.extractor.run(prompt)
        (run_dir / "01_entities.json").write_text(
            extraction.model_dump_json(indent=2), encoding="utf-8"
        )

        constraint_spec = self.lookup.run(extraction)
        (run_dir / "02_constraint_spec.json").write_text(
            constraint_spec.model_dump_json(indent=2), encoding="utf-8"
        )

        reports: List[VerificationReport] = []
        remediation_hints: List[str] = []
        final_image_path = run_dir / "image_attempt_1.png"

        for attempt in range(1, self.max_retries + 2):
            image_path = run_dir / f"image_attempt_{attempt}.png"
            generated_path, grounded_prompt = self.generator.run(
                spec=constraint_spec,
                output_path=image_path,
                remediation_hints=remediation_hints or None,
            )
            (run_dir / f"grounded_prompt_attempt_{attempt}.txt").write_text(
                grounded_prompt, encoding="utf-8"
            )

            report = self.verifier.run(
                spec=constraint_spec,
                image_path=generated_path,
                attempt=attempt,
            )
            (run_dir / f"verification_report_attempt_{attempt}.json").write_text(
                report.model_dump_json(indent=2), encoding="utf-8"
            )
            (run_dir / f"verification_report_attempt_{attempt}.md").write_text(
                report.to_markdown(), encoding="utf-8"
            )

            reports.append(report)
            final_image_path = generated_path

            if report.overall_passed:
                break

            remediation_hints = [
                v.remediation_hint or f"Fix failed check: {v.criterion}"
                for v in report.failed_checks()
            ]

        return PipelineRunResult(
            run_dir=run_dir,
            extraction=extraction,
            constraint_spec=constraint_spec,
            reports=reports,
            final_image_path=final_image_path,
            final_report=reports[-1],
        )
