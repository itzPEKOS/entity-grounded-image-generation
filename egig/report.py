"""Markdown reports (one per attempt + a final summary)."""

from __future__ import annotations

from typing import List

from .schemas import AttemptReport, ConstraintSpec, GenerationRecord, Status

ICON = {Status.passed: "PASS", Status.failed: "**FAIL**", Status.skipped: "skip"}


def _cell(text: str) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def attempt_markdown(report: AttemptReport, record: GenerationRecord, spec: ConstraintSpec) -> str:
    v = report.verdict
    lines = [
        f"# Attempt {report.attempt}: {'PASS' if v.passed else 'FAIL'}",
        "",
        f"- Prompt sent to the generator: `{_cell(record.prompt)}`",
        f"- Negative prompt: `{_cell(record.negative_prompt)}`",
        f"- Backend: {record.backend}, seed {record.params.seed}, "
        f"ip_scale {record.params.ip_scale}, control_scale {record.params.control_scale:.2f}",
        "",
        "## How the verdict was computed",
        "",
        f"1. Critical checks: {v.critical_passed}/{v.critical_total} passed "
        f"-> {'all passed' if v.critical_passed == v.critical_total else 'at least one failed'}",
        f"2. Major checks: {v.major_passed}/{v.major_total} = {v.major_rate:.0%}",
        f"3. Rule: all critical pass AND major rate >= threshold -> **{'PASS' if v.passed else 'FAIL'}**",
        f"   ({_cell(v.explanation)})",
    ]
    if v.skipped:
        lines.append(f"4. {v.skipped} check(s) skipped (not counted)")
    if report.vlm_error:
        lines += ["", f"> **Gemini was unavailable**, so the VLM checks below are marked FAIL as *not verified* "
                      f"(the image may be fine). Error: {_cell(report.vlm_error[:300])}"]
    lines += ["", "## Checks", "", "| Check | Severity | Method | Result | Conf. | Evidence |", "|---|---|---|---|---|---|"]
    for r in report.results:
        conf = f"{r.confidence:.2f}" if r.confidence is not None else ""
        lines.append(f"| `{r.check_id}` {_cell(r.description)} | {r.severity.value} | {r.method.value} | "
                     f"{ICON[r.status]} | {conf} | {_cell(r.evidence)} |")
    if report.ref_similarity:
        lines += ["", "## Similarity to reference photos (CLIP cosine, info only)", ""]
        lines += [f"- {k}: {s:.3f}" for k, s in report.ref_similarity.items()]
    if report.detections:
        lines += ["", "## Detections (Grounding DINO)", ""]
        lines += [f"- {d.entity_id}: '{d.label}' score {d.score:.2f} box {d.box}"
                  f"{' (low confidence)' if d.low_confidence else ''}" for d in report.detections]
    if report.vlm_counts:
        lines += ["", "## Gemini's own count (second opinion for count checks)", ""]
        lines += [f"- {k}: {n}" for k, n in report.vlm_counts.items()]
    failed = report.failed()
    if failed:
        lines += ["", "## Failed checks: advice from the verifier (for people; the generator acts on the check "
                      "type, see the repair log of the next attempt)", ""]
        lines += [f"- `{r.check_id}`: {_cell(r.fix_hint or r.evidence)}" for r in failed]
    if record.params.notes:
        lines += ["", "## What the generator changed for this attempt (repair log)", ""]
        lines += [f"- {n}" for n in record.params.notes]
    return "\n".join(lines) + "\n"


def final_markdown(spec: ConstraintSpec, reports: List[AttemptReport], records: List[GenerationRecord],
                   best_index: int, llm_calls: int) -> str:
    best = reports[best_index]
    lines = [
        f"# Final result: {'PASS' if best.verdict.passed else 'FAIL'} (attempt {best.attempt})",
        "",
        f"Request: *{spec.prompt}*",
        "",
        "| Attempt | Verdict | Critical | Major | Seed | ip_scale | control |",
        "|---|---|---|---|---|---|---|",
    ]
    for rep, rec in zip(reports, records):
        v = rep.verdict
        verdict = "PASS" if v.passed else ("FAIL (not verified)" if rep.vlm_error else "FAIL")
        lines.append(f"| {rep.attempt} | {verdict} | {v.critical_passed}/{v.critical_total} | "
                     f"{v.major_passed}/{v.major_total} ({v.major_rate:.0%}) | {rec.params.seed} | "
                     f"{rec.params.ip_scale} | {rec.params.control_scale:.2f} |")
    unverified = [r.attempt for r in reports if r.vlm_error]
    if unverified:
        lines += ["", f"> **Gemini was unavailable** during verification of attempt(s) {unverified}, so their "
                      "Gemini checks are counted as FAIL *without looking at the image*, and the loop stopped "
                      "(repairing from checks nobody made would only waste GPU time). Run again later. "
                      f"Error: {_cell(reports[-1].vlm_error[:300] if reports[-1].vlm_error else '')}"]
    if not best.verdict.passed:
        lines += ["", "No attempt passed; the best one is returned (most critical checks passed, "
                      "then highest major rate, then highest reference similarity)."]
    lines += ["", "## Entities and their references", ""]
    for e in spec.entities:
        lines.append(f"- **{e.name}** (`{e.id}`, {e.category.value}, detector label '{e.detection_label}', "
                     f"{'rigid' if e.rigid else 'non-rigid'}): {len(e.references)} reference photo(s)")
        for url in e.source_urls[:4]:
            lines.append(f"  - {url}")
    lines += ["", f"Gemini calls used in this run: {llm_calls}"]
    return "\n".join(lines) + "\n"
