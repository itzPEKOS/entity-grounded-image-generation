import json

import httpx

from egig.backends.collage import CollageBackend
from egig.orchestrator import build_orchestrator
from egig.wiki import WikiClient

from conftest import FakeBackend, FakeLLM, FakeWiki, profile_for, vlm_answer


def make(settings, parsed_query, vlm_answers, vision, backend=None):
    llm = FakeLLM({"ParsedQuery": parsed_query, "VisualProfile": profile_for, "VlmVerdicts": vlm_answers})
    orch = build_orchestrator(settings, llm=llm, backend=backend or FakeBackend(), vision=vision, wiki=FakeWiki())
    return orch, llm


def test_fail_then_pass(settings, parsed_query, plan_matching_vision):
    orch, llm = make(settings, parsed_query, [vlm_answer(fail=("shiba_inu.attr1",)), vlm_answer()],
                     plan_matching_vision)
    result = orch.run(parsed_query.english_prompt, run_id="t1")

    assert [r.verdict.passed for r in result.reports] == [False, True]
    assert result.passed and result.best_index == 1
    # feedback reached the generator: dog's reference strength lowered for attempt 2
    assert result.records[0].params.ip_scale["shiba_inu"] == 0.6
    assert result.records[1].params.ip_scale["shiba_inu"] == 0.45
    assert "Shiba Inu sitting" in result.records[1].prompt
    # Gemini calls: 1 parse + 2 profiles + 2 verifications = 5
    assert llm.calls == 5

    run = result.run_dir
    for name in ("01_spec.json", "02_spec_grounded.json", "03_checks.json", "trace.jsonl", "final.png",
                 "final_report.md", "attempt_1/report.md", "attempt_1/image.png", "attempt_1/control_canny.png",
                 "attempt_2/detections.png"):
        assert (run / name).exists(), name
    trace = [json.loads(line) for line in (run / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ("Verifier", "ImageGenerator") in [(t["from"], t["to"]) for t in trace]
    grounded = json.loads((run / "02_spec_grounded.json").read_text(encoding="utf-8"))
    assert all(len(e["references"]) == 3 for e in grounded["entities"])  # lead + 2 commons, all detected
    assert "repair log" in (run / "attempt_2" / "report.md").read_text(encoding="utf-8")


def test_all_attempts_fail_returns_best(settings, parsed_query, plan_matching_vision):
    answers = [vlm_answer(fail=("scene.1", "shiba_inu.attr1")), vlm_answer(fail=("scene.1",)),
               vlm_answer(fail=("scene.1", "relation.1", "shiba_inu.attr1"))]
    orch, _ = make(settings, parsed_query, answers, plan_matching_vision)
    result = orch.run("x", run_id="t2")
    assert len(result.reports) == 3 and not result.passed
    assert result.best_index == 1  # attempt 2 failed the fewest critical checks
    assert "No attempt passed" in (result.run_dir / "final_report.md").read_text(encoding="utf-8")


def test_collage_backend_runs_the_whole_loop(settings, parsed_query, plan_matching_vision):
    orch, _ = make(settings, parsed_query, [vlm_answer()], plan_matching_vision, backend=CollageBackend())
    result = orch.run("x", run_id="t3")
    assert result.passed and result.final_image.exists()


def test_reference_crops_rigid_union_vs_animal_best_box(settings, parsed_query, tmp_path):
    from egig.agents.parser import normalize
    from egig.agents.retriever import EntityRetrieverAgent
    from conftest import FakeVision

    vision = FakeVision({"temple": [(0.8, [0.4, 0.1, 0.6, 0.5]), (0.5, [0.1, 0.3, 0.25, 0.6])],
                         "dog": [(0.9, [0.2, 0.2, 0.5, 0.9]), (0.4, [0.6, 0.5, 0.7, 0.6])]})
    llm = FakeLLM({"VisualProfile": profile_for})
    agent = EntityRetrieverAgent(llm, FakeWiki(), vision, settings)
    spec = agent.run(normalize("x", parsed_query), tmp_path / "refs")
    temple_crop = spec.entity("wat_arun").references[0].crop_box
    dog_crop = spec.entity("shiba_inu").references[0].crop_box
    assert temple_crop == [0.06, 0.06, 0.64, 0.64]  # union [0.1, 0.1, 0.6, 0.6] grown by 0.04
    assert dog_crop == [0.16, 0.16, 0.54, 0.94]  # best dog box [0.2, 0.2, 0.5, 0.9] grown by 0.04


def test_wiki_client_parsing(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        host, params = request.url.host, dict(request.url.params)
        if host == "en.wikipedia.org":
            assert params["gsrsearch"] == "Mercury"
            return httpx.Response(200, json={"query": {"pages": [
                {"index": 1, "title": "Mercury", "pageprops": {"disambiguation": ""}},
                {"index": 2, "title": "Mercury (planet)", "fullurl": "https://en.wikipedia.org/wiki/Mercury_(planet)",
                 "extract": "Mercury is the first planet.", "thumbnail": {"source": "https://up.test/m.jpg"}},
            ]}})
        if host == "commons.wikimedia.org":
            return httpx.Response(200, json={"query": {"pages": [
                {"index": 2, "title": "File:b.jpg", "imageinfo": [{"mime": "image/jpeg", "width": 1200, "height": 800,
                                                                   "thumburl": "https://up.test/b.jpg"}]},
                {"index": 1, "title": "File:a.svg", "imageinfo": [{"mime": "image/svg+xml", "width": 2000}]},
                {"index": 3, "title": "File:tiny.jpg", "imageinfo": [{"mime": "image/jpeg", "width": 100}]},
            ]}})
        return httpx.Response(404)

    wiki = WikiClient("test-agent", http=httpx.Client(transport=httpx.MockTransport(handler)))
    page = wiki.find_page("Mercury")
    assert page.title == "Mercury (planet)" and page.image_url == "https://up.test/m.jpg"  # disambiguation skipped
    imgs = wiki.commons_images("Mercury")
    assert [i.url for i in imgs] == ["https://up.test/b.jpg"]  # svg and tiny images filtered out
    assert wiki.download_image("https://up.test/missing.jpg", tmp_path / "x.jpg") is None
