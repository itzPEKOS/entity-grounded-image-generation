from pathlib import Path

from PIL import Image

from egig.agents.parser import normalize
from egig.agents.verifier import VerificationAgent, assign_detections, build_checks, compute_verdict
from egig.schemas import (CheckKind, CheckResult, Method, Severity, Status, VisualProfile, VlmVerdict,
                          VlmVerdicts)
from egig.vision import RawDetection

from conftest import FakeLLM, FakeVision, vlm_answer


def spec_with_profiles(parsed_query):
    spec = normalize(parsed_query.english_prompt, parsed_query)
    spec.entity("wat_arun").profile = VisualProfile(summary="", key_features=["prang", "mosaics", "4 prangs", "stairs"],
                                                    negative_traits=[], confusables=["Wat Pho"], prompt_keywords="")
    spec.entity("shiba_inu").profile = VisualProfile(summary="", key_features=["ears", "tail", "urajiro"],
                                                     negative_traits=[], confusables=[], prompt_keywords="")
    return spec


def test_build_checks(parsed_query):
    checks = build_checks(spec_with_profiles(parsed_query))
    ids = [c.id for c in checks]
    # wat_arun: presence, position, identity, 4 features = 7; shiba: presence, position, identity, attr1, 3 features = 7
    # + scene.1 + relation.1 = 16
    assert len(ids) == 16
    assert "shiba_inu.attr1" in ids and "scene.1" in ids and "relation.1" in ids
    sev = {c.id: c.severity for c in checks}
    assert sev["shiba_inu.attr1"] == Severity.critical and sev["wat_arun.feat2"] == Severity.major
    assert sev["shiba_inu.position"] == Severity.major


def _r(sev, status):
    return CheckResult(check_id="c", kind=CheckKind.feature, severity=sev, method=Method.vlm, description="",
                       status=status)


def test_verdict_rule_step_by_step():
    crit_ok = [_r(Severity.critical, Status.passed)] * 5
    major_3_of_4 = [_r(Severity.major, Status.passed)] * 3 + [_r(Severity.major, Status.failed)]
    v = compute_verdict(crit_ok + major_3_of_4, 0.7)
    assert v.passed and v.major_rate == 0.75  # 3/4 = 75% >= 70%

    v = compute_verdict(crit_ok[:4] + [_r(Severity.critical, Status.failed)] + major_3_of_4, 0.7)
    assert not v.passed and v.critical_passed == 4  # one critical fail is enough to FAIL

    major_2_of_4 = [_r(Severity.major, Status.passed)] * 2 + [_r(Severity.major, Status.failed)] * 2
    assert not compute_verdict(crit_ok + major_2_of_4, 0.7).passed  # 2/4 = 50% < 70%

    skipped = [_r(Severity.critical, Status.skipped)] * 3
    v = compute_verdict(crit_ok + skipped, 0.7)
    assert v.passed and v.critical_total == 5 and v.skipped == 3  # skips are not counted


def test_two_entities_with_the_same_label_get_their_own_detection(parsed_query):
    spec = normalize("two dogs", parsed_query)
    a, b = spec.entities
    a.detection_label = b.detection_label = "dog"
    a.box, b.box = [0.0, 0.4, 0.45, 1.0], [0.55, 0.4, 1.0, 1.0]
    raw = {"dog": [RawDetection("dog", 0.8, [0.6, 0.5, 0.95, 0.95]), RawDetection("dog", 0.7, [0.05, 0.5, 0.4, 0.95])]}
    assigned = assign_detections(spec.entities, raw)
    assert assigned[a.id][0].score == 0.7 and assigned[b.id][0].score == 0.8


def test_verifier_end_to_end_with_fakes(parsed_query, settings, plan_matching_vision, tmp_path):
    spec = spec_with_profiles(parsed_query)
    checks = build_checks(spec)
    img_path = tmp_path / "img.png"
    Image.new("RGB", (256, 256), (10, 20, 30)).save(img_path)

    def vlm_skipping_one(parts):
        full = vlm_answer(fail=("shiba_inu.attr1",))(parts)
        return VlmVerdicts(verdicts=[v for v in full.verdicts if v.check_id != "scene.1"])

    agent = VerificationAgent(FakeLLM({"VlmVerdicts": vlm_skipping_one}), plan_matching_vision, settings)
    report = agent.run(spec, checks, img_path, attempt=1, out_dir=tmp_path)
    status = {r.check_id: r.status for r in report.results}

    assert status["wat_arun.presence"] == Status.passed and status["shiba_inu.position"] == Status.passed
    assert status["shiba_inu.attr1"] == Status.failed
    assert status["scene.1"] == Status.failed  # VLM forgot it -> counted as FAIL, not silently passed
    assert not report.verdict.passed
    assert (tmp_path / "detections.png").exists()


def test_missing_entity_and_wrong_position(parsed_query, settings, tmp_path):
    spec = spec_with_profiles(parsed_query)
    img_path = tmp_path / "img.png"
    Image.new("RGB", (256, 256)).save(img_path)
    vision = FakeVision({"temple": [(0.6, [0.0, 0.75, 0.2, 1.0])]})  # temple bottom-left, dog missing
    agent = VerificationAgent(FakeLLM({"VlmVerdicts": vlm_answer()}), vision, settings)
    report = agent.run(spec, build_checks(spec), img_path, 1, tmp_path)
    status = {r.check_id: r.status for r in report.results}
    assert status["shiba_inu.presence"] == Status.failed
    assert status["wat_arun.position"] == Status.failed


def test_rigid_entity_position_uses_union_of_its_parts(parsed_query, settings, tmp_path):
    # Temple found as two separate towers at the far left and far right edges.
    # Best box alone: centre x = 0.015, outside the planned box even with the 0.1 margin (starts at 0.02) -> would FAIL.
    # Union of both: [0.0, 0.1, 1.0, 0.6], centre (0.5, 0.35), inside -> PASS.
    spec = spec_with_profiles(parsed_query)
    img_path = tmp_path / "img.png"
    Image.new("RGB", (100, 100)).save(img_path)
    vision = FakeVision({"temple": [(0.8, [0.0, 0.1, 0.03, 0.6]), (0.7, [0.9, 0.1, 1.0, 0.6])],
                         "dog": [(0.9, [0.35, 0.5, 0.65, 0.95])]})
    report = VerificationAgent(FakeLLM({"VlmVerdicts": vlm_answer()}), vision, settings).run(
        spec, build_checks(spec), img_path, 1, tmp_path)
    pos = next(r for r in report.results if r.check_id == "wat_arun.position")
    assert pos.status == Status.passed and "union of 2 boxes" in pos.evidence


def test_without_detector_detector_checks_are_skipped(parsed_query, settings, tmp_path):
    spec = spec_with_profiles(parsed_query)
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64)).save(img_path)
    report = VerificationAgent(FakeLLM({"VlmVerdicts": vlm_answer()}), None, settings).run(
        spec, build_checks(spec), img_path, 1, tmp_path)
    skipped = {r.check_id for r in report.results if r.status == Status.skipped}
    assert skipped == {"wat_arun.presence", "wat_arun.position", "shiba_inu.presence", "shiba_inu.position"}
    assert report.verdict.passed


# ------------------------------------------------- count cross-check (penguin runs)
from egig.agents.verifier import drop_group_boxes, score_tuple  # noqa: E402
from egig.schemas import AttemptReport, Verdict, VlmCount  # noqa: E402

LEFT, MID, RIGHT = [0.1, 0.4, 0.35, 0.95], [0.38, 0.4, 0.62, 0.95], [0.65, 0.4, 0.9, 0.95]


def penguin_spec(parsed_query):
    spec = normalize("Three penguins standing in a row", parsed_query)
    p = spec.entity("shiba_inu")
    p.count, p.detection_label, p.attributes = 3, "penguin", ["in a row"]
    p.box, p.profile, p.references = [0.05, 0.35, 0.95, 0.98], None, []
    spec.entities, spec.relations, spec.scene.requirements = [p], [], []
    return spec


def verify_penguins(parsed_query, settings, tmp_path, boxes, gemini_count):
    spec = penguin_spec(parsed_query)

    def answer(parts):
        reply = vlm_answer()(parts)
        if gemini_count is not None:
            reply.counts = [VlmCount(entity_id="shiba_inu", count=gemini_count)]
        return reply

    agent = VerificationAgent(FakeLLM({"VlmVerdicts": answer}), FakeVision({"penguin": boxes}), settings)
    image = tmp_path / "img.png"
    Image.new("RGB", (64, 64)).save(image)
    report = agent.run(spec, build_checks(spec), image, 1, tmp_path)
    return report, {r.check_id: r for r in report.results}


def test_count_unsure_detector_uses_gemini_count(parsed_query, settings, tmp_path):
    # like attempt 3: three penguins in a row, the two behind score below 0.35
    report, r = verify_penguins(parsed_query, settings, tmp_path,
                                [(0.44, LEFT), (0.31, MID), (0.27, RIGHT)], gemini_count=3)
    assert r["shiba_inu.count"].status == Status.passed
    assert "used Gemini's count 3" in r["shiba_inu.count"].evidence
    assert report.verdict.passed and report.vlm_counts == {"shiba_inu": 3}
    assert sum(d.low_confidence for d in report.detections) == 2


def test_count_unsure_detector_and_gemini_sees_four(parsed_query, settings, tmp_path):
    # like attempt 2: two big penguins + blurry ones behind; Gemini counts 4
    _, r = verify_penguins(parsed_query, settings, tmp_path,
                           [(0.41, LEFT), (0.35, RIGHT), (0.27, MID)], gemini_count=4)
    assert r["shiba_inu.count"].status == Status.failed


def test_count_conflict_when_detector_is_consistent(parsed_query, settings, tmp_path):
    _, r = verify_penguins(parsed_query, settings, tmp_path, [(0.44, LEFT)], gemini_count=3)
    assert r["shiba_inu.count"].status == Status.failed and "CONFLICT" in r["shiba_inu.count"].evidence


def test_count_without_gemini_falls_back_to_detector(parsed_query, settings, tmp_path):
    _, r = verify_penguins(parsed_query, settings, tmp_path,
                           [(0.44, LEFT), (0.31, MID), (0.27, RIGHT)], gemini_count=None)
    assert r["shiba_inu.count"].status == Status.failed and "no Gemini count" in r["shiba_inu.count"].evidence


def test_presence_passes_with_low_confidence_box(parsed_query, settings, tmp_path):
    _, r = verify_penguins(parsed_query, settings, tmp_path, [(0.29, MID)], gemini_count=1)
    assert r["shiba_inu.presence"].status == Status.passed and "low confidence" in r["shiba_inu.presence"].evidence


def test_group_box_around_all_penguins_is_dropped(parsed_query, settings, tmp_path):
    group = RawDetection(label="penguin", score=0.5, box=[0.05, 0.35, 0.95, 0.97])
    singles = [RawDetection(label="penguin", score=0.45, box=b) for b in (LEFT, MID, RIGHT)]
    assert drop_group_boxes([group] + singles) == singles
    _, r = verify_penguins(parsed_query, settings, tmp_path,
                           [(0.5, group.box), (0.45, LEFT), (0.45, MID), (0.45, RIGHT)], gemini_count=3)
    assert r["shiba_inu.count"].evidence == "found 3, expected 3"


def test_best_attempt_prefers_the_request_over_clip_similarity():
    def rep(attempt, attr_conf, clip):
        res = [CheckResult(check_id="p.attr1", kind=CheckKind.attribute, severity=Severity.critical,
                           method=Method.vlm, description="", status=Status.passed, confidence=attr_conf)]
        v = Verdict(passed=False, critical_passed=4, critical_total=5, major_passed=5, major_total=5,
                    major_rate=1.0, skipped=0, explanation="")
        return AttemptReport(attempt=attempt, image_path="x", results=res, verdict=v, ref_similarity={"p": clip})
    # penguin run: attempt 1 looked more like the reference (CLIP 0.767) but was less clearly "in a row"
    one, three = rep(1, 0.90, 0.767), rep(3, 1.00, 0.691)
    assert max([one, three], key=score_tuple) is three
