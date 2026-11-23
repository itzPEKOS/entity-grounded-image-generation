"""Agent 4 - Verification Evaluator.

Two layers of checks:
1. Grounding DINO (detector): is every entity there, how many, and is it where the layout
   planned it? These are measured with boxes and coordinates.
2. Gemini (VLM): is it really *that* entity (compared side by side with the reference
   photo), does it have the user's requested attributes, its key visual features, the scene
   requirements and the relations?

PASS/FAIL is computed here in code from the individual results, never taken from the VLM.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from ..config import Settings
from ..geometry import center, contains_point, expand, iou, to_pixels, union_box
from ..llm import LLM, LLMError
from ..schemas import (AttemptReport, Check, CheckKind, CheckResult, ConstraintSpec, Detection, EntitySpec,
                       Method, Severity, Status, Verdict, VlmVerdicts)
from ..vision import RawDetection, VisionTools, cosine

SYSTEM = """You are the Verification agent of an image-generation pipeline. Be strict and literal.
Judge each check ONLY from what is visible in IMAGE A (the generated image). Reference photos show
what the real entity looks like; crops are zoomed-in parts of IMAGE A.
Return one verdict per check_id, using the check_id exactly as given.
- passed: true only if the check is clearly satisfied in IMAGE A.
- confidence: 0..1.
- evidence: one short sentence about what you actually see.
- fix_hint: if failed, a short instruction (max 15 words) telling the image generator what to change; else ""."""


# ------------------------------------------------------------------------- checks
def build_checks(spec: ConstraintSpec) -> List[Check]:
    checks: List[Check] = []

    def add(cid: str, entity: Optional[str], kind: CheckKind, sev: Severity, method: Method, desc: str) -> None:
        checks.append(Check(id=cid, entity_id=entity, kind=kind, severity=sev, method=method, description=desc))

    for e in spec.entities:
        add(f"{e.id}.presence", e.id, CheckKind.presence, Severity.critical, Method.detector,
            f"{e.name}: the detector finds a '{e.detection_label}'")
        if e.count > 1:
            add(f"{e.id}.count", e.id, CheckKind.count, Severity.critical, Method.detector,
                f"{e.name}: exactly {e.count} x '{e.detection_label}'")
        add(f"{e.id}.position", e.id, CheckKind.position, Severity.major, Method.detector,
            f"{e.name}: centre of the detected box lies inside the planned box {e.box}")
        if e.needs_reference and (e.references or e.profile):
            add(f"{e.id}.identity", e.id, CheckKind.identity, Severity.critical, Method.vlm,
                f"{e.name}: the {e.detection_label} is recognizably {e.name} (same as the reference photo), "
                f"not a look-alike{' such as ' + ', '.join(e.profile.confusables) if e.profile and e.profile.confusables else ''}")
        for i, attr in enumerate(e.attributes, 1):
            add(f"{e.id}.attr{i}", e.id, CheckKind.attribute, Severity.critical, Method.vlm, f"{e.name}: {attr}")
        for i, feat in enumerate(e.profile.key_features if e.profile else [], 1):
            add(f"{e.id}.feat{i}", e.id, CheckKind.feature, Severity.major, Method.vlm, f"{e.name}: {feat}")
    for i, req in enumerate(spec.scene.requirements, 1):
        add(f"scene.{i}", None, CheckKind.scene, Severity.critical, Method.vlm, f"Scene: {req}")
    for i, rel in enumerate(spec.relations, 1):
        s, o = spec.entity(rel.subject), spec.entity(rel.object)
        if s and o:
            add(f"relation.{i}", None, CheckKind.relation, Severity.critical, Method.vlm,
                f"Relation: {s.name} {rel.predicate} {o.name}")
    return checks


# ------------------------------------------------------------------------ verdict
def compute_verdict(results: List[CheckResult], major_pass_rate: float) -> Verdict:
    """PASS = every critical check passed AND at least `major_pass_rate` of major checks
    passed. Skipped checks are left out of both counts."""
    crit = [r for r in results if r.severity == Severity.critical and r.status != Status.skipped]
    major = [r for r in results if r.severity == Severity.major and r.status != Status.skipped]
    crit_ok = sum(r.status == Status.passed for r in crit)
    major_ok = sum(r.status == Status.passed for r in major)
    rate = major_ok / len(major) if major else 1.0
    passed = crit_ok == len(crit) and rate >= major_pass_rate
    explanation = (
        f"critical {crit_ok}/{len(crit)} passed ({'all' if crit_ok == len(crit) else 'NOT all'}); "
        f"major {major_ok}/{len(major)} = {rate:.0%} (need >= {major_pass_rate:.0%}) -> {'PASS' if passed else 'FAIL'}"
    )
    return Verdict(passed=passed, critical_passed=crit_ok, critical_total=len(crit), major_passed=major_ok,
                   major_total=len(major), major_rate=round(rate, 4),
                   skipped=sum(r.status == Status.skipped for r in results), explanation=explanation)


def score_tuple(report: AttemptReport) -> Tuple[float, float, float]:
    """For picking the best attempt when none passes."""
    v = report.verdict
    crit = v.critical_passed / v.critical_total if v.critical_total else 1.0
    sims = list(report.ref_similarity.values())
    return (crit, v.major_rate, sum(sims) / len(sims) if sims else 0.0)


# --------------------------------------------------------------------- detections
def entity_extent(e: EntitySpec, dets: List[RawDetection]) -> List[float]:
    """Where the entity is: a rigid landmark is often detected as several parts (each tower of
    a temple), so use the union of its boxes; for anything else use the best box."""
    return union_box([d.box for d in dets]) if e.rigid else dets[0].box


def assign_detections(entities: List[EntitySpec], raw: Dict[str, List[RawDetection]]) -> Dict[str, List[RawDetection]]:
    """Entities can share a detector label (two dogs). Give each detection to the entity
    whose planned box overlaps it most (ties: nearest planned centre)."""
    out: Dict[str, List[RawDetection]] = {e.id: [] for e in entities}
    by_label: Dict[str, List[EntitySpec]] = defaultdict(list)
    for e in entities:
        by_label[e.detection_label].append(e)
    for label, group in by_label.items():
        for det in raw.get(label, []):
            cx, cy = center(det.box)

            def closeness(e: EntitySpec) -> Tuple[float, float]:
                ex, ey = center(e.box)
                return (iou(det.box, e.box), -((cx - ex) ** 2 + (cy - ey) ** 2))

            out[max(group, key=closeness).id].append(det)
    for dets in out.values():
        dets.sort(key=lambda d: d.score, reverse=True)
    return out


class VerificationAgent:
    name = "Verifier"

    def __init__(self, llm: LLM, vision: Optional[VisionTools], settings: Settings):
        self.llm = llm
        self.vision = vision
        self.s = settings

    def run(self, spec: ConstraintSpec, checks: List[Check], image_path: Path, attempt: int,
            out_dir: Path) -> AttemptReport:
        image = Image.open(image_path).convert("RGB")
        results: Dict[str, CheckResult] = {}
        detections: List[Detection] = []
        crops: Dict[str, Image.Image] = {}
        ref_similarity: Dict[str, float] = {}
        det_checks = [c for c in checks if c.method == Method.detector]

        if self.vision is None:
            for c in det_checks:
                results[c.id] = CheckResult.of(c, Status.skipped, "detector not available (install torch + transformers)")
        else:
            labels = sorted({e.detection_label for e in spec.entities})
            raw = {label: self.vision.detect(image, label) for label in labels}
            assigned = assign_detections(spec.entities, raw)
            for e in spec.entities:
                dets = assigned[e.id]
                detections += [Detection(entity_id=e.id, label=d.label, score=round(d.score, 3),
                                         box=[round(v, 3) for v in d.box]) for d in dets]
                if dets:
                    crops[e.id] = image.crop(to_pixels(expand(entity_extent(e, dets), 0.03), image.width, image.height))
            for c in det_checks:
                results[c.id] = self._detector_result(c, spec.entity(c.entity_id), assigned)
            ref_similarity = self._ref_similarity(spec, crops)

        vlm_checks = [c for c in checks if c.method == Method.vlm]
        vlm_error = ""
        if vlm_checks:
            try:
                results.update(self._ask_vlm(spec, vlm_checks, image, crops))
            except LLMError as err:
                # Count as FAIL, not SKIP: a skipped check is left out of the verdict, so an image
                # nobody looked at could otherwise PASS on the detector checks alone.
                vlm_error = str(err)
                print(f"[verifier] Gemini unavailable: {len(vlm_checks)} checks not verified")
                for c in vlm_checks:
                    results[c.id] = CheckResult.of(c, Status.failed, "NOT VERIFIED: Gemini was unavailable")

        ordered = [results[c.id] for c in checks]
        verdict = compute_verdict(ordered, self.s.major_pass_rate)
        draw_detections(image, spec, detections).save(out_dir / "detections.png")
        return AttemptReport(attempt=attempt, image_path=str(image_path), results=ordered, detections=detections,
                             ref_similarity=ref_similarity, verdict=verdict, vlm_error=vlm_error)

    # ------------------------------------------------------------- detector part
    def _detector_result(self, c: Check, e: Optional[EntitySpec], assigned: Dict[str, List[RawDetection]]) -> CheckResult:
        assert e is not None
        dets = assigned[e.id]
        if c.kind == CheckKind.presence:
            if dets:
                return CheckResult.of(c, Status.passed, f"found '{e.detection_label}' (score {dets[0].score:.2f})",
                                      confidence=dets[0].score)
            return CheckResult.of(c, Status.failed, f"no '{e.detection_label}' above score {self.s.box_threshold}",
                                  fix_hint=f"make the {e.name} clearly visible")
        if c.kind == CheckKind.count:
            ok = len(dets) == e.count
            return CheckResult.of(c, Status.passed if ok else Status.failed,
                                  f"found {len(dets)}, expected {e.count}")
        # position
        if not dets:
            return CheckResult.of(c, Status.failed, "entity not detected, so its position cannot match")
        extent = entity_extent(e, dets)
        cx, cy = center(extent)
        ok = contains_point(e.box, (cx, cy), margin=self.s.position_margin)
        evidence = (f"detected {'union of ' + str(len(dets)) + ' boxes' if e.rigid and len(dets) > 1 else 'box'} "
                    f"{[round(v, 2) for v in extent]} centre ({cx:.2f}, {cy:.2f}); "
                    f"planned {e.box} (+{self.s.position_margin} margin); IoU {iou(extent, e.box):.2f}")
        return CheckResult.of(c, Status.passed if ok else Status.failed, evidence)

    def _ref_similarity(self, spec: ConstraintSpec, crops: Dict[str, Image.Image]) -> Dict[str, float]:
        """CLIP cosine similarity between the generated crop and its reference crops (best of).
        Reported as a number for the report; it does not decide PASS/FAIL."""
        sims: Dict[str, float] = {}
        assert self.vision is not None
        for e in spec.entities:
            if e.id not in crops or not e.references:
                continue
            refs = [Image.open(r.crop_path).convert("RGB") for r in e.references]
            embs = self.vision.image_embeddings([crops[e.id]] + refs)
            sims[e.id] = round(max(cosine(embs[0], r) for r in embs[1:]), 4)
        return sims

    # ------------------------------------------------------------------ VLM part
    def _ask_vlm(self, spec: ConstraintSpec, checks: List[Check], image: Image.Image,
                 crops: Dict[str, Image.Image]) -> Dict[str, CheckResult]:
        parts: list = [f"User request: {spec.prompt!r}", "IMAGE A (generated image to verify):", image]
        for e in spec.entities:
            if e.references:
                parts += [f"REFERENCE photo of the real {e.name}:", Image.open(e.references[0].crop_path)]
            if e.id in crops:
                parts += [f"CROP of IMAGE A where the {e.detection_label} ({e.name}) was detected:", crops[e.id]]
        lines = "\n".join(f"- check_id={c.id} | {c.description}" for c in checks)
        parts.append(f"Checks:\n{lines}")

        reply = self.llm.generate_json(VlmVerdicts, parts, system=SYSTEM, vision=True)
        by_id = {v.check_id.strip(): v for v in reply.verdicts}
        out: Dict[str, CheckResult] = {}
        for c in checks:
            v = by_id.get(c.id)
            if v is None:  # the VLM skipped it: count as failed rather than silently passing
                out[c.id] = CheckResult.of(c, Status.failed, "VLM returned no verdict for this check")
                continue
            out[c.id] = CheckResult.of(
                c, Status.passed if v.passed else Status.failed, v.evidence.strip(),
                confidence=round(min(max(v.confidence, 0.0), 1.0), 3),
                fix_hint="" if v.passed else v.fix_hint.strip(),
            )
        return out


def draw_detections(image: Image.Image, spec: ConstraintSpec, detections: List[Detection]) -> Image.Image:
    """Planned boxes (thin, grey) and detected boxes (thick, coloured) on the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    palette = [(230, 57, 70), (29, 120, 220), (42, 157, 80), (240, 160, 0), (130, 60, 200)]
    colors = {e.id: palette[i % len(palette)] for i, e in enumerate(spec.entities)}
    for e in spec.entities:
        draw.rectangle(to_pixels(e.box, img.width, img.height), outline=(200, 200, 200), width=2)
    for d in detections:
        color = colors.get(d.entity_id or "", (255, 255, 255))
        px = to_pixels(d.box, img.width, img.height)
        draw.rectangle(px, outline=color, width=5)
        draw.text((px[0] + 6, px[1] + 6), f"{d.entity_id} {d.score:.2f}", fill=color)
    return img
