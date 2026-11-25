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
from ..geometry import area, center, contains_point, expand, intersection, iou, to_pixels, union_box
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
- fix_hint: if failed, a short instruction (max 15 words) telling the image generator what to change; else "".
- visible: false when the part a check is about cannot be seen at all (outside the image, cropped off,
  hidden behind something), e.g. "curled tail" on a head-and-shoulders picture. Then passed=false.
  A feature that IS visible but looks wrong is visible=true, passed=false.
Actions, poses and relations must be visibly happening, not just plausible from the context:
e.g. a dog standing next to a ball is not "playing with the ball"; a cup beside a saucer is not "on the saucer".
Also return `counts`: one entry per entity_id listed under "Counts", with how many you see."""


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
        else:  # one asked for: extra copies are a defect, but the request is still there -> major
            add(f"{e.id}.count", e.id, CheckKind.count, Severity.major, Method.detector,
                f"{e.name}: only one (no extra copies), by Gemini's count")
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
    if spec.scene.setting.strip():
        # major: Agent 1 sometimes infers the setting, and place names ("Chiang Rai") cannot be seen
        add("scene.setting", None, CheckKind.scene, Severity.major, Method.vlm,
            f"Scene: the place looks like {spec.scene.setting.strip()} (judge what is visible: landscape, "
            f"vegetation, buildings; an exact place name cannot be confirmed)")
    for i, req in enumerate(spec.scene.requirements, 1):
        add(f"scene.{i}", None, CheckKind.scene, Severity.critical, Method.vlm, f"Scene: {req}")
    add("image.integrity", None, CheckKind.integrity, Severity.critical, Method.vlm,
        "Image: the whole image is one continuous scene. It is NOT shown as a framed picture, poster, "
        "photo-inside-a-photo or collage, and has no border or panel around or across it "
        "(ordinary frames hanging on a wall inside the scene are fine)")
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


REQUEST_KINDS = (CheckKind.count, CheckKind.attribute, CheckKind.relation, CheckKind.scene)


def score_tuple(report: AttemptReport) -> Tuple[float, float, float, float, float]:
    """For picking the best attempt when none passes, in this order:
    1. share of critical checks passed
    2. share of the user's explicit requests passed (count, attributes, relations, scene)
    3. share of major checks passed
    4. mean confidence on the passed requests (Gemini's certainty)
    5. CLIP similarity to the reference photos (looks like the real thing; last, because a
       photo-like copy of the reference can score high while ignoring the request)."""
    v = report.verdict
    crit = v.critical_passed / v.critical_total if v.critical_total else 1.0
    req = [r for r in report.results if r.kind in REQUEST_KINDS and r.status != Status.skipped]
    req_ok = [r for r in req if r.status == Status.passed]
    req_rate = len(req_ok) / len(req) if req else 1.0
    confs = [r.confidence for r in req_ok if r.confidence is not None]
    sims = list(report.ref_similarity.values())
    return (crit, req_rate, v.major_rate, sum(confs) / len(confs) if confs else 0.0,
            sum(sims) / len(sims) if sims else 0.0)


# --------------------------------------------------------------------- detections
def entity_extent(e: EntitySpec, dets: List[RawDetection]) -> List[float]:
    """Where the entity is: a rigid landmark is often detected as several parts (each tower of
    a temple), so use the union of its boxes; for anything else use the best box."""
    return union_box([d.box for d in dets]) if e.rigid else dets[0].box


def drop_group_boxes(dets: List[RawDetection], inside: float = 0.85) -> List[RawDetection]:
    """Grounding DINO often adds one big box around a whole group ("the penguins") on top of the
    individual boxes. A box that contains two or more smaller boxes is that group box: drop it,
    otherwise three penguins are counted as four."""
    keep = []
    for d in dets:
        inner = sum(1 for o in dets if o is not d and area(o.box) < area(d.box)
                    and area(intersection(o.box, d.box)) >= inside * area(o.box))
        if inner < 2:
            keep.append(d)
    return keep


def drop_boxes_spanning_others(entities: List[EntitySpec],
                               assigned: Dict[str, List[RawDetection]]) -> Dict[str, List[RawDetection]]:
    """Cat + dog + rabbit run: the animals merged into one creature and the detector put the same
    big box around it for "cat", "dog" and "rabbit", so every presence check passed while the
    rabbit was gone. A box for a non-rigid entity that also covers the planned centres of two or
    more other entities on its layer is a box around a group, not around this entity: ignore it.
    (Rigid entities keep theirs: a temple legitimately spans the whole picture.)"""
    out: Dict[str, List[RawDetection]] = {}
    for e in entities:
        others = [center(o.box) for o in entities if o is not e and o.layer == e.layer]
        if e.rigid or len(others) < 2:
            out[e.id] = assigned.get(e.id, [])
            continue
        out[e.id] = [d for d in assigned.get(e.id, [])
                     if sum(contains_point(d.box, c) for c in others) < 2]
    return out


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

        sure: Dict[str, List[RawDetection]] = {e.id: [] for e in spec.entities}  # score >= box_threshold
        loose: Dict[str, List[RawDetection]] = {e.id: [] for e in spec.entities}  # score >= box_threshold_low
        if self.vision is not None:
            # one detector pass at the low threshold; the normal threshold is a filter on top
            # how many things each label should find: group boxes only matter when it is more than one
            # (a single temple is legitimately one big box around its towers)
            expected: Dict[str, int] = defaultdict(int)
            for e in spec.entities:
                expected[e.detection_label] += e.count
            raw_loose = {}
            for label, n in sorted(expected.items()):
                dets = self.vision.detect(image, label, box_threshold=self.s.box_threshold_low)
                raw_loose[label] = drop_group_boxes(dets) if n > 1 else dets
            raw_sure = {label: [d for d in dets if d.score >= self.s.box_threshold] for label, dets in raw_loose.items()}
            sure = drop_boxes_spanning_others(spec.entities, assign_detections(spec.entities, raw_sure))
            loose = drop_boxes_spanning_others(spec.entities, assign_detections(spec.entities, raw_loose))
            for e in spec.entities:
                for d in loose[e.id]:
                    detections.append(Detection(entity_id=e.id, label=d.label, score=round(d.score, 3),
                                                box=[round(v, 3) for v in d.box],
                                                low_confidence=d.score < self.s.box_threshold))
                dets = sure[e.id] or loose[e.id]
                if dets:
                    crops[e.id] = image.crop(to_pixels(expand(entity_extent(e, dets), 0.03), image.width, image.height))
            ref_similarity = self._ref_similarity(spec, crops)

        vlm_checks = [c for c in checks if c.method == Method.vlm]
        vlm_error = ""
        vlm_counts: Dict[str, int] = {}
        if vlm_checks:
            try:
                vlm_results, vlm_counts = self._ask_vlm(spec, vlm_checks, image, crops)
                results.update(vlm_results)
            except LLMError as err:
                # Count as FAIL, not SKIP: a skipped check is left out of the verdict, so an image
                # nobody looked at could otherwise PASS on the detector checks alone.
                vlm_error = str(err)
                print(f"[verifier] Gemini unavailable: {len(vlm_checks)} checks not verified")
                for c in vlm_checks:
                    results[c.id] = CheckResult.of(c, Status.failed, "NOT VERIFIED: Gemini was unavailable")

        # detector checks last, so a count can use Gemini's count as a second opinion
        for c in det_checks:
            if self.vision is None:
                results[c.id] = CheckResult.of(c, Status.skipped, "detector not available (install torch + transformers)")
            else:
                results[c.id] = self._detector_result(c, spec.entity(c.entity_id), sure, loose, vlm_counts)

        ordered = [results[c.id] for c in checks]
        verdict = compute_verdict(ordered, self.s.major_pass_rate)
        if vlm_error:
            verdict.explanation += f" [Gemini unavailable: {len(vlm_checks)} checks NOT VERIFIED]"
        draw_detections(image, spec, detections).save(out_dir / "detections.png")
        return AttemptReport(attempt=attempt, image_path=str(image_path), results=ordered, detections=detections,
                             ref_similarity=ref_similarity, verdict=verdict, vlm_error=vlm_error,
                             vlm_counts=vlm_counts)

    # ------------------------------------------------------------- detector part
    def _detector_result(self, c: Check, e: Optional[EntitySpec], sure: Dict[str, List[RawDetection]],
                         loose: Dict[str, List[RawDetection]], vlm_counts: Dict[str, int]) -> CheckResult:
        assert e is not None
        hi, lo = self.s.box_threshold, self.s.box_threshold_low
        if c.kind == CheckKind.presence:
            if vlm_counts.get(e.id) == 0:
                # second opinion: Gemini looked for this entity and saw none (merged or missing animal),
                # so a detector box (often one box around a merged creature) does not prove it is there
                found = sure[e.id] or loose[e.id]
                detector = f"detector found '{e.detection_label}' (score {found[0].score:.2f})" if found \
                    else f"detector found no '{e.detection_label}'"
                return CheckResult.of(c, Status.failed, f"Gemini counted 0; {detector}",
                                      fix_hint=f"make the {e.name} clearly visible")
            if sure[e.id]:
                d = sure[e.id][0]
                return CheckResult.of(c, Status.passed, f"found '{e.detection_label}' (score {d.score:.2f})",
                                      confidence=d.score)
            if loose[e.id]:
                d = loose[e.id][0]
                return CheckResult.of(c, Status.passed, f"found '{e.detection_label}' only with low confidence "
                                      f"(score {d.score:.2f}, between {lo} and {hi})", confidence=d.score)
            note = f"; Gemini counted {vlm_counts[e.id]}" if vlm_counts.get(e.id) else ""
            return CheckResult.of(c, Status.failed, f"no '{e.detection_label}' above score {lo}{note}",
                                  fix_hint=f"make the {e.name} clearly visible")
        if c.kind == CheckKind.count:
            return self._count_result(c, e, len(sure[e.id]), len(loose[e.id]), vlm_counts.get(e.id))
        # position
        dets = sure[e.id] or loose[e.id]
        if not dets:
            return CheckResult.of(c, Status.failed, "entity not detected, so its position cannot match")
        extent = entity_extent(e, dets)
        cx, cy = center(extent)
        ok = contains_point(e.box, (cx, cy), margin=self.s.position_margin)
        evidence = (f"detected {'union of ' + str(len(dets)) + ' boxes' if e.rigid and len(dets) > 1 else 'box'} "
                    f"{[round(v, 2) for v in extent]} centre ({cx:.2f}, {cy:.2f}); "
                    f"planned {e.box} (+{self.s.position_margin} margin); IoU {iou(extent, e.box):.2f}")
        return CheckResult.of(c, Status.passed if ok else Status.failed, evidence)

    def _count_result(self, c: Check, e: EntitySpec, n_sure: int, n_loose: int, n_vlm: Optional[int]) -> CheckResult:
        """Cross-check the count.
        - detector count at the normal threshold matches -> PASS
        - detector is UNSURE (different counts at the two thresholds: overlapping or half-hidden
          animals score low) -> Gemini's count decides
        - detector is sure but Gemini disagrees -> FAIL, marked as CONFLICT in the evidence
        - no Gemini count (unavailable) -> detector decides, as before"""
        n = e.count
        hi, lo = self.s.box_threshold, self.s.box_threshold_low
        if n == 1:
            # one asked for: the detector often splits one landmark or train into several boxes, so
            # only Gemini's count is used, to catch extra copies (two scooters when one was asked for)
            if n_vlm is None:
                return CheckResult.of(c, Status.skipped, "no Gemini count")
            if n_vlm == 1:
                return CheckResult.of(c, Status.passed, "Gemini counted 1")
            if n_vlm == 0:
                return CheckResult.of(c, Status.failed, f"Gemini counted 0 (detector found {n_sure})",
                                      fix_hint=f"make the {e.name} clearly visible")
            return CheckResult.of(c, Status.failed, f"Gemini counted {n_vlm}, only one was asked for",
                                  fix_hint=f"show a single {e.detection_label}")
        if n_sure == n:
            return CheckResult.of(c, Status.passed, f"found {n_sure}, expected {n}")
        detector = f"detector found {n_sure} at score >= {hi} and {n_loose} at >= {lo}"
        if n_vlm is None:
            return CheckResult.of(c, Status.failed, f"found {n_sure}, expected {n} ({detector}; no Gemini count)",
                                  fix_hint=f"show exactly {n} {e.detection_label}s, clearly separated")
        if n_sure != n_loose:
            ok = n_vlm == n
            return CheckResult.of(c, Status.passed if ok else Status.failed,
                                  f"{detector} (unsure) -> used Gemini's count {n_vlm}, expected {n}",
                                  confidence=0.6, fix_hint="" if ok else f"show exactly {n} {e.detection_label}s")
        if n_vlm == n:
            return CheckResult.of(c, Status.failed, f"CONFLICT: {detector}, but Gemini counted {n_vlm}; "
                                  f"expected {n}. Counted as FAIL because the detector was consistent",
                                  fix_hint=f"show exactly {n} {e.detection_label}s, clearly separated")
        return CheckResult.of(c, Status.failed, f"found {n_sure}, expected {n} (Gemini counted {n_vlm})",
                              fix_hint=f"show exactly {n} {e.detection_label}s, clearly separated")

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
                 crops: Dict[str, Image.Image]) -> Tuple[Dict[str, CheckResult], Dict[str, int]]:
        parts: list = [f"User request: {spec.prompt!r}", "IMAGE A (generated image to verify):", image]
        for e in spec.entities:
            if e.references:
                parts += [f"REFERENCE photo of the real {e.name}:", Image.open(e.references[0].crop_path)]
            if e.id in crops:
                parts += [f"CROP of IMAGE A where the {e.detection_label} ({e.name}) was detected:", crops[e.id]]
        count_lines = "\n".join(f"- entity_id={e.id} | {e.name} ({e.detection_label})" for e in spec.entities)
        parts.append("Counts: for each entity below, count how many are visible in IMAGE A, including partly "
                     "hidden, small or blurry background ones. Count what you see; do not copy the number from "
                     f"the request.\n{count_lines}")
        lines = "\n".join(f"- check_id={c.id} | {c.description}" for c in checks)
        parts.append(f"Checks:\n{lines}")

        reply = self.llm.generate_json(VlmVerdicts, parts, system=SYSTEM, vision=True)
        known = {e.id for e in spec.entities}
        counts = {k.entity_id.strip(): max(int(k.count), 0) for k in reply.counts if k.entity_id.strip() in known}
        by_id = {v.check_id.strip(): v for v in reply.verdicts}
        out: Dict[str, CheckResult] = {}
        for c in checks:
            v = by_id.get(c.id)
            if v is None:  # the VLM skipped it: count as failed rather than silently passing
                out[c.id] = CheckResult.of(c, Status.failed, "VLM returned no verdict for this check")
                continue
            if c.kind == CheckKind.feature and not v.visible and not v.passed:
                # a key feature that is simply not in view (no tail in a portrait) is not a mistake;
                # user requests (attributes, identity, scene) still have to be visible
                out[c.id] = CheckResult.of(c, Status.skipped, "not visible: " + v.evidence.strip())
                continue
            out[c.id] = CheckResult.of(
                c, Status.passed if v.passed else Status.failed, v.evidence.strip(),
                confidence=round(min(max(v.confidence, 0.0), 1.0), 3),
                fix_hint="" if v.passed else v.fix_hint.strip(),
            )
        return out, counts


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
        draw.rectangle(px, outline=color, width=2 if d.low_confidence else 5)  # thin = low confidence
        draw.text((px[0] + 6, px[1] + 6), f"{d.entity_id} {d.score:.2f}{' (low)' if d.low_confidence else ''}",
                  fill=color)
    return img
