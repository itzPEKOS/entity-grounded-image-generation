import numpy as np
from PIL import Image

from egig.agents.generator import TOKEN_BUDGET, _fit, build_negative, build_prompt
from egig.agents.parser import normalize
from egig.backends import estimate_tokens
from egig.conditioning import control_image, fit_into_box, layer_masks
from egig.geometry import to_pixels
from egig.repair import repair
from egig.schemas import (AttemptReport, CheckKind, CheckResult, GenParams, Method, Severity, Status, Verdict,
                          VisualProfile)

from conftest import striped_image


def grounded_spec(parsed_query):
    spec = normalize(parsed_query.english_prompt, parsed_query)
    wat, shiba = spec.entities
    wat.profile = VisualProfile(summary="", key_features=["tall central prang"], negative_traits=["golden chedi"],
                                confusables=[], prompt_keywords="Wat Arun prang, porcelain mosaic")
    shiba.profile = VisualProfile(summary="", key_features=["curled tail"], negative_traits=["floppy ears"],
                                  confusables=[], prompt_keywords="red Shiba Inu, curled tail")
    return spec


# --------------------------------------------------------------------- masks / control
def test_foreground_cuts_its_area_out_of_the_background_mask(parsed_query):
    spec = grounded_spec(parsed_query)
    masks = layer_masks(spec.entities, 200, 200)
    x0, y0, x1, y1 = to_pixels(spec.entity("shiba_inu").box, 200, 200)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    assert masks["shiba_inu"].getpixel((cx, cy)) == 255
    assert masks["wat_arun"].getpixel((cx, 50)) == 255  # temple area above the dog: still the temple
    assert masks["wat_arun"].getpixel((cx, cy)) == 0  # behind the dog: removed from the temple


def test_fully_hidden_entity_keeps_its_box(parsed_query):
    spec = grounded_spec(parsed_query)
    spec.entity("shiba_inu").box = [0.0, 0.0, 1.0, 1.0]  # dog covers everything
    masks = layer_masks(spec.entities, 100, 100)
    assert masks["wat_arun"].getbbox() is not None


def test_fit_into_box_keeps_aspect_and_sits_on_bottom():
    img = Image.new("RGB", (400, 200))  # 2:1
    resized, (ox, oy) = fit_into_box(img, (0, 0, 100, 100), align="bottom")
    assert resized.size == (100, 50) and (ox, oy) == (0, 50)


def test_control_image_only_draws_rigid_entities_and_clears_foreground(parsed_query):
    spec = grounded_spec(parsed_query)
    crops = {"wat_arun": striped_image("temple"), "shiba_inu": striped_image("dog")}
    ctrl = np.asarray(control_image(spec.entities, crops, 256, 256))[:, :, 0]
    assert ctrl.max() == 255  # temple edges are there
    x0, y0, x1, y1 = to_pixels(spec.entity("shiba_inu").box, 256, 256)
    assert ctrl[y0:y1, x0:x1].max() == 0  # dog box cleared: the dog is free to take any pose
    wx0, wy0, wx1, wy1 = to_pixels(spec.entity("wat_arun").box, 256, 256)
    outside = ctrl.copy()
    outside[wy0:wy1, wx0:wx1] = 0
    assert outside.max() == 0  # nothing drawn outside the temple box


def test_control_image_is_none_without_rigid_references(parsed_query):
    spec = grounded_spec(parsed_query)
    assert control_image(spec.entities, {"shiba_inu": striped_image("dog")}, 64, 64) is None


# ---------------------------------------------------------------------------- prompts
def test_prompt_starts_with_user_request_and_respects_budget(parsed_query):
    spec = grounded_spec(parsed_query)
    params = GenParams(seed=1, extra_prompt=["Shiba Inu sitting"])
    prompt = build_prompt(spec, params, estimate_tokens)
    assert prompt.startswith("A Shiba Inu sitting in front of Wat Arun at sunset, Shiba Inu sitting")
    assert "Wat Arun prang" in prompt and estimate_tokens(prompt) <= TOKEN_BUDGET


def test_fit_drops_low_priority_segments_and_truncates_first():
    count = lambda s: len(s.replace(",", " ").split())  # 1 token per word
    assert _fit(["a b c", "d e", "f g h i"], count, budget=6) == "a b c, d e"
    assert _fit(["a b c d e f g h", "x"], count, budget=4) == "a b c d"


def test_negative_never_forbids_what_the_user_asked_for(parsed_query):
    spec = grounded_spec(parsed_query)
    spec.entity("shiba_inu").attributes = ["with floppy ears"]
    negative = build_negative(spec, GenParams(seed=1), estimate_tokens)
    assert "floppy ears" not in negative and "golden chedi" in negative and "blurry" in negative


# ----------------------------------------------------------------------------- repair
def report_with(*failed):
    results = [CheckResult(check_id=cid, entity_id=eid, kind=kind, severity=Severity.critical, method=Method.vlm,
                           description=desc, status=Status.failed, evidence="", fix_hint=hint)
               for cid, eid, kind, desc, hint in failed]
    verdict = Verdict(passed=False, critical_passed=0, critical_total=len(results), major_passed=0, major_total=0,
                      major_rate=1.0, skipped=0, explanation="")
    return AttemptReport(attempt=1, image_path="x.png", results=results, verdict=verdict)


def test_attribute_failure_lowers_ip_scale_identity_failure_raises_it(parsed_query):
    spec = grounded_spec(parsed_query)
    params = GenParams(seed=10, ip_scale={"shiba_inu": 0.6, "wat_arun": 0.6}, control_scale=0.5)
    rep = report_with(
        ("shiba_inu.attr1", "shiba_inu", CheckKind.attribute, "Shiba Inu: sitting", "make the dog sit"),
        ("wat_arun.identity", "wat_arun", CheckKind.identity, "Wat Arun: recognizably Wat Arun", ""),
    )
    new = repair(params, rep, spec)
    assert new.seed == 11
    assert new.ip_scale == {"shiba_inu": 0.45, "wat_arun": 0.75}  # 0.60-0.15 and 0.60+0.15
    assert new.emphasis == {"shiba_inu": 1}
    assert new.extra_prompt == ["Shiba Inu sitting"]  # rule-based phrase; Gemini's free-text hint is not pasted
    assert "golden chedi" in new.extra_negative
    assert any("ip_scale[shiba_inu] 0.60 -> 0.45" in n for n in new.notes)
    assert params.ip_scale["shiba_inu"] == 0.6  # original untouched


def test_scene_failure_lowers_every_reference_and_scales_are_bounded(parsed_query):
    spec = grounded_spec(parsed_query)
    params = GenParams(seed=1, ip_scale={"shiba_inu": 0.32, "wat_arun": 0.9}, control_scale=0.5)
    new = repair(params, report_with(("scene.1", None, CheckKind.scene, "Scene: at sunset", "")), spec)
    assert new.ip_scale == {"shiba_inu": 0.3, "wat_arun": 0.85}  # 0.32-0.05 clamped to the 0.30 floor
    assert new.extra_prompt == ["at sunset"]
