"""Evaluation tooling: batch runner survives errors, saved runs can be re-verified and compared with labels."""

import csv
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from egig.evaluation import (ReplayVerifier, compare, find_runs, load_prompts, load_run, localize_refs,
                             read_labels, replay_loop, reverify_run, run_batch)
from egig.llm import LLMError
from egig.orchestrator import build_orchestrator
from egig.schemas import AttemptReport, Verdict

from conftest import FakeBackend, FakeLLM, FakeWiki, profile_for, vlm_answer

ROOT = Path(__file__).resolve().parents[1]


def report(passed: bool, critical=(1, 1), vlm_error="") -> AttemptReport:
    return AttemptReport(attempt=1, image_path="x.png", results=[], vlm_error=vlm_error,
                         verdict=Verdict(passed=passed, critical_passed=critical[0], critical_total=critical[1],
                                         major_passed=1, major_total=1, major_rate=1.0, skipped=0, explanation=""))


# ----------------------------------------------------------------- prompt set
def test_prompt_set_is_valid_and_split_between_two_people():
    path = ROOT / "eval" / "prompts.csv"
    dev = load_prompts(path, split="dev")
    assert len(dev) == 15
    assert len(load_prompts(path, split="dev", part="A")) == 8
    assert len(load_prompts(path, split="dev", part="B")) == 7
    assert len(load_prompts(path, split="holdout")) == 8
    assert all(p.prompt and p.id.isascii() for p in load_prompts(path))  # ids become folder names
    assert [p.id for p in load_prompts(path, only=["d03_shiba_akita"])] == ["d03_shiba_akita"]
    assert any("วัดอรุณ" in p.prompt for p in dev)  # Thai survives the CSV


# ----------------------------------------------------------------- batch on Kaggle
class FakeOrch:
    """orch.run() stand-in: `plan` maps prompt -> list of outcomes, one per call
    ("pass" / "fail" / "notverified" / an exception)."""

    def __init__(self, tmp_path, plan):
        self.s = SimpleNamespace(output_dir=str(tmp_path))
        self.plan = plan
        self.calls = []

    def run(self, prompt, run_id=None):
        self.calls.append(prompt)
        run_dir = Path(self.s.output_dir) / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "partial.txt").write_text("x")
        what = self.plan[prompt].pop(0)
        if isinstance(what, Exception):
            raise what
        rep = report(what == "pass", vlm_error="503 overloaded" if what == "notverified" else "")
        return SimpleNamespace(reports=[rep], best_index=0, llm_calls=4, run_dir=run_dir)


def prompts(*names):
    from egig.evaluation import EvalPrompt
    return [EvalPrompt(id=n, prompt=n, part="A") for n in names]


def test_batch_keeps_going_when_a_prompt_errors(tmp_path):
    orch = FakeOrch(tmp_path, {
        "ok": ["pass"],
        "flaky": [LLMError("Gemini error (HTTP 503)"), "fail"],  # error, then works on the retry
        "dead": [RuntimeError("CUDA out of memory"), RuntimeError("CUDA out of memory")],
        "late": ["notverified", "pass"],
    })
    waits = []
    rows = run_batch(orch, prompts("ok", "flaky", "dead", "late"), summary_name="summary_A.csv",
                     commit="abc123", sleep=waits.append, retry_wait_s=5)

    assert {k: r["status"] for k, r in rows.items()} == {"ok": "PASS", "flaky": "FAIL", "dead": "ERROR", "late": "PASS"}
    assert waits == [5, 5, 5]  # one wait before each retry
    assert "CUDA out of memory" in rows["dead"]["error"]
    summary = list(csv.DictReader(open(tmp_path / "eval" / "summary_A.csv", encoding="utf-8")))
    assert [r["id"] for r in summary] == ["dead", "flaky", "late", "ok"]
    assert all(r["commit"] == "abc123" for r in summary)


def test_batch_rerun_skips_finished_prompts(tmp_path):
    orch = FakeOrch(tmp_path, {"ok": ["pass"], "dead": [RuntimeError("x"), RuntimeError("x"), "fail"]})
    run_batch(orch, prompts("ok", "dead"), sleep=lambda s: None)
    rows = run_batch(orch, prompts("ok", "dead"), sleep=lambda s: None)  # same cell again
    assert orch.calls == ["ok", "dead", "dead", "dead"]  # "ok" was not run twice
    assert rows["dead"]["status"] == "FAIL"


def test_batch_waits_longer_when_gemini_seems_down(tmp_path):
    err = lambda: LLMError("503")  # noqa: E731
    orch = FakeOrch(tmp_path, {n: [err(), err()] for n in "abc"} | {"d": ["pass"]})
    waits = []
    run_batch(orch, prompts("a", "b", "c", "d"), sleep=waits.append, retry_wait_s=1, outage_wait_s=99,
              max_consecutive_errors=3)
    assert waits == [1, 1, 1, 99]


def test_generator_only_batch_keeps_not_verified_images(tmp_path):
    # max_attempts=1: the picture is what is tested; Gemini being down only means it is verified later
    orch = FakeOrch(tmp_path, {"a": ["notverified"], "b": ["fail"]})
    waits = []
    rows = run_batch(orch, prompts("a", "b"), sleep=waits.append, rerun_not_verified=False)
    rows = run_batch(orch, prompts("a", "b"), sleep=waits.append, rerun_not_verified=False)  # same cell again
    assert orch.calls == ["a", "b"] and waits == []  # no retry, nothing generated twice
    assert rows["a"]["status"] == "NOT_VERIFIED"


# ----------------------------------------------------------------- replaying the loop
def test_replay_loop_stops_like_the_pipeline():
    assert replay_loop([report(True), report(False)], 3) == {"best_index": 0, "status": "PASS", "used": 1,
                                                               "incomplete": False}
    r = replay_loop([report(False, (0, 2)), report(False, (1, 2))], 3)
    assert r["best_index"] == 1 and r["status"] == "FAIL" and r["incomplete"]  # attempt 3 was never made
    r = replay_loop([report(False, vlm_error="503")], 3)
    assert r["status"] == "NOT_VERIFIED" and not r["incomplete"]


def test_localize_refs_points_kaggle_paths_at_the_local_folder(tmp_path, parsed_query):
    from egig.schemas import ConstraintSpec, EntitySpec, ReferenceImage
    ref = ReferenceImage(path="/kaggle/working/outputs/eval_dev/d01/refs/wat_arun/cand_00.jpg",
                         crop_path="/kaggle/working/outputs/eval_dev/d01/refs/wat_arun/cand_00_crop.jpg",
                         url="u", source="commons")
    entity = EntitySpec(**parsed_query.entities[0].model_dump(), references=[ref])
    spec = ConstraintSpec(prompt="p", scene=parsed_query.scene, entities=[entity])
    local = localize_refs(spec, tmp_path / "d01")
    assert Path(local.entities[0].references[0].crop_path) == tmp_path / "d01" / "refs" / "wat_arun" / "cand_00_crop.jpg"
    assert spec.entities[0].references[0].crop_path.startswith("/kaggle")  # the original is not changed


# ----------------------------------------------------------------- reverify + report on a real (fake-model) run
@pytest.fixture
def saved_run(settings, parsed_query, plan_matching_vision):
    llm = FakeLLM({"ParsedQuery": parsed_query, "VisualProfile": profile_for,
                   "VlmVerdicts": [vlm_answer(fail=("shiba_inu.attr1",)), vlm_answer()]})
    orch = build_orchestrator(settings, llm=llm, backend=FakeBackend(), vision=plan_matching_vision, wiki=FakeWiki())
    result = orch.run(parsed_query.english_prompt, run_id="eval_dev/d02_shiba_watarun")
    return result.run_dir


def test_reverify_with_saved_gemini_answers_reproduces_the_verdicts(saved_run, settings, plan_matching_vision, tmp_path):
    def no_gemini(*a, **k):
        raise AssertionError("Gemini must not be called with reuse_vlm")
    llm = SimpleNamespace(calls=0, generate_json=no_gemini)
    verifier = ReplayVerifier(llm, plan_matching_vision, settings, reuse_vlm=True)
    new = reverify_run(saved_run, tmp_path / "v2" / saved_run.name, verifier)

    old = load_run(saved_run)
    assert [r.verdict.passed for r in new.reports] == [r.verdict.passed for r in old.reports] == [False, True]
    assert verifier.reused == 2
    assert load_run(saved_run, tmp_path / "v2" / saved_run.name).reports[1].verdict.passed


def test_reverify_with_a_changed_detector_changes_the_verdict(saved_run, settings, tmp_path):
    from conftest import FakeVision
    llm = FakeLLM({"VlmVerdicts": vlm_answer()})
    blind = FakeVision({})  # the new detector finds nothing -> presence checks fail
    new = reverify_run(saved_run, tmp_path / "v2", ReplayVerifier(llm, blind, settings))
    assert not any(r.verdict.passed for r in new.reports)
    assert llm.calls == 2  # reuse_vlm off: Gemini asked again for both images
    assert replay_loop(new.reports, 3)["incomplete"]  # attempt 3 would have been generated


def test_compare_and_report(saved_run, tmp_path, monkeypatch):
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("run_id,attempt,human,tags,note,labeler\n"
                          "d02_shiba_watarun,1,ok,,looks sitting to me,me\n"
                          "d02_shiba_watarun,2,wrong,identity;scene,,me\n", encoding="utf-8")
    labels = read_labels([labels_csv])
    rows = compare([load_run(d) for d in find_runs(saved_run.parent)], labels, 3)
    assert [a.outcome for a in rows[0].attempts] == ["false FAIL", "false PASS"]
    assert rows[0].status == "PASS" and rows[0].final_attempt == 2

    out = tmp_path / "report.md"
    monkeypatch.setattr(sys, "argv", ["eval_report.py", str(saved_run.parent), "--labels", str(labels_csv),
                                      "--out", str(out)])
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(ROOT / "scripts" / "eval_report.py"), run_name="__main__")
    assert done.value.code == 0
    text = out.read_text(encoding="utf-8")
    assert "false PASS: 1" in text and "false FAIL: 1" in text and "| identity | 1 |" in text


def test_review_page_lists_every_attempt_image(saved_run, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["make_review.py", str(saved_run.parent)])
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(ROOT / "scripts" / "make_review.py"), run_name="__main__")
    assert done.value.code == 0
    page = (saved_run.parent / "review.html").read_text(encoding="utf-8")
    assert "d02_shiba_watarun/attempt_2/image.png" in page
    assert "PASS" not in page and "not as requested" not in page  # the system's verdict is not shown
    assert "refs/" in page  # reference photos shown to the labeller


def test_label_of_an_older_picture_is_ignored(saved_run, tmp_path):
    """A prompt run again keeps its folder name and attempt numbers, but the pictures are new."""
    from egig.evaluation import image_sig
    sig = image_sig(saved_run / "attempt_1" / "image.png")
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("run_id,attempt,human,tags,note,labeler,image\n"
                          f"d02_shiba_watarun,1,ok,,,me,{sig}\n"
                          "d02_shiba_watarun,2,ok,,,me,0123456789\n", encoding="utf-8")
    rows = compare([load_run(saved_run)], read_labels([labels_csv]), 3)
    assert rows[0].attempts[0].label is not None  # same picture
    assert rows[0].attempts[1].label is None  # judged some other picture


def test_resume_redoes_only_the_images_gemini_could_not_check(saved_run, settings, plan_matching_vision, tmp_path):
    out = tmp_path / "v2" / saved_run.name
    calls = []

    def flaky(parts):  # image 1 checked, image 2 hits a Gemini outage
        calls.append(1)
        if len(calls) == 2:
            raise LLMError("503")
        return vlm_answer()(parts)
    down = FakeLLM({"VlmVerdicts": flaky})
    first = reverify_run(saved_run, out, ReplayVerifier(down, plan_matching_vision, settings))
    assert [bool(r.vlm_error) for r in first.reports] == [False, True]

    up = FakeLLM({"VlmVerdicts": vlm_answer()})
    second = reverify_run(saved_run, out, ReplayVerifier(up, plan_matching_vision, settings), resume=True)
    assert up.calls == 1  # only image 2 asked again
    assert not any(r.vlm_error for r in second.reports)
