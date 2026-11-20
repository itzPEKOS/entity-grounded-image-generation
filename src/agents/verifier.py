"""Agent 4: Vision-Language Auditor that checks the generated image against every constraint."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.models import (
    CheckVerdict,
    GlobalConstraintSpec,
    VerificationReport,
)


class VerificationAgent:
    """Evaluates generated images against every check in GlobalConstraintSpec."""

    def __init__(
        self,
        vlm_client: Optional[object] = None,
        model: str = "gemini-3.8-flash",
        pass_threshold: float = 1.0,
    ):
        self.vlm_client = vlm_client
        self.model = model
        self.pass_threshold = pass_threshold

    def run(
        self,
        spec: GlobalConstraintSpec,
        image_path: Path,
        attempt: int = 1,
    ) -> VerificationReport:
        if self.vlm_client is not None:
            return self._verify_with_vlm(spec, image_path, attempt)
        return self._verify_demo(spec, image_path, attempt)

    def _verify_with_vlm(
        self,
        spec: GlobalConstraintSpec,
        image_path: Path,
        attempt: int,
    ) -> VerificationReport:
        from google.genai import types
        from src.gemini_helper import generate_with_fallback

        checks_payload = [c.model_dump() for c in spec.all_checks()]
        prompt = (
            "Inspect the attached generated image strictly against each visual check below. "
            "For every check_id, decide passed (true/false), provide a confidence score (0-1), "
            "describe concrete visual evidence observed in the image, and if failed, give a concrete "
            "remediation_hint for regenerating the image.\n\n"
            f"Checks to evaluate:\n{checks_payload}"
        )
        image_part = types.Part.from_bytes(
            data=image_path.read_bytes(),
            mime_type="image/png",
        )
        response = generate_with_fallback(
            client=self.vlm_client,
            primary_model=self.model,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                system_instruction="You are a strict Visual Verification Auditor Agent.",
                response_mime_type="application/json",
                response_schema=VerificationReport,
            ),
        )
        report = VerificationReport.model_validate_json(response.text)
        report.attempt = attempt
        report.image_path = str(image_path)
        return report

    def _verify_demo(
        self,
        spec: GlobalConstraintSpec,
        image_path: Path,
        attempt: int,
    ) -> VerificationReport:
        all_checks = spec.all_checks()
        verdicts: List[CheckVerdict] = []

        for idx, chk in enumerate(all_checks):
            simulated_fail = attempt == 1 and idx == 1 and len(all_checks) > 2
            passed = not simulated_fail
            verdicts.append(
                CheckVerdict(
                    check_id=chk.check_id,
                    entity_id=chk.entity_id,
                    criterion=chk.criterion,
                    passed=passed,
                    confidence=0.94 if passed else 0.88,
                    observed_evidence=(
                        f"Verified visual presence of '{chk.criterion}' in rendered output."
                        if passed
                        else f"Fine-grained detail '{chk.criterion}' was under-emphasized in attempt 1."
                    ),
                    remediation_hint=(
                        None
                        if passed
                        else f"Explicitly foreground and sharpen: {chk.criterion}"
                    ),
                )
            )

        passed_count = sum(1 for v in verdicts if v.passed)
        pass_rate = passed_count / max(len(verdicts), 1)
        overall_passed = pass_rate >= self.pass_threshold

        return VerificationReport(
            attempt=attempt,
            image_path=str(image_path),
            overall_passed=overall_passed,
            pass_rate=pass_rate,
            verdicts=verdicts,
            summary=(
                "All entity visual constraints and global composition checks passed."
                if overall_passed
                else f"{len(verdicts) - passed_count} check(s) failed; triggering grounded prompt refinement."
            ),
        )
