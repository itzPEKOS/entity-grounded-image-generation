from egig.agents.parser import QueryParserAgent, normalize
from egig.geometry import clamp_box, contains_point, iou, nms
from egig.schemas import Category, ParsedEntity, Relation
from egig.vision import RawDetection

from conftest import FakeLLM


def test_clamp_box_fixes_order_range_and_size():
    assert clamp_box([0.9, 1.2, -0.1, 0.5]) == [0.0, 0.5, 0.9, 1.0]
    tiny = clamp_box([0.5, 0.5, 0.51, 0.51], min_size=0.2)
    assert round(tiny[2] - tiny[0], 3) == 0.2 and round(tiny[3] - tiny[1], 3) == 0.2


def test_iou_by_hand():
    # overlap 0.25 x 0.25 = 0.0625; union 0.25 + 0.25 - 0.0625 = 0.4375; 0.0625 / 0.4375 = 0.142857
    assert abs(iou([0, 0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]) - 0.142857) < 1e-5
    assert iou([0, 0, 0.1, 0.1], [0.5, 0.5, 0.6, 0.6]) == 0.0


def test_nms_keeps_best_of_overlapping_boxes():
    dets = [RawDetection("dog", 0.5, [0.1, 0.1, 0.5, 0.5]), RawDetection("dog", 0.9, [0.12, 0.1, 0.5, 0.52]),
            RawDetection("dog", 0.6, [0.6, 0.6, 0.9, 0.9])]
    kept = nms(dets, 0.5)
    assert [d.score for d in kept] == [0.9, 0.6]


def test_contains_point_with_margin():
    assert not contains_point([0.3, 0.3, 0.6, 0.6], (0.65, 0.5))
    assert contains_point([0.3, 0.3, 0.6, 0.6], (0.65, 0.5), margin=0.1)


def test_normalize_cleans_llm_output(parsed_query):
    extra = [ParsedEntity(id="Shiba Inu", name="Shiba Inu", category=Category.animal, search_query="",
                          detection_label="Dog.", count=99, attributes=[" running ", "running", ""], box=[0.1],
                          layer=7, needs_reference=True, rigid=False)]
    extra += [parsed_query.entities[0].model_copy(update={"id": f"x{i}"}) for i in range(3)]  # 6 entities total
    parsed = parsed_query.model_copy(update={
        "entities": parsed_query.entities + extra,
        "relations": parsed_query.relations + [Relation(subject="ghost", predicate="near", object="wat_arun")],
    })
    spec = normalize("ภาพ Shiba Inu หน้าวัดอรุณ", parsed)

    assert len(spec.entities) == 4  # trimmed to MAX_ENTITIES
    ids = [e.id for e in spec.entities]
    assert ids[:3] == ["wat_arun", "shiba_inu", "shiba_inu_2"]  # duplicate id gets a suffix
    dup = spec.entities[2]
    assert dup.count == 6 and dup.layer == 2 and dup.detection_label == "dog"
    assert dup.attributes == ["running"]
    assert dup.search_query == "Shiba Inu"
    assert len(dup.box) == 4  # invalid box replaced by a default layout box
    assert [r.subject for r in spec.relations] == ["shiba_inu"]  # unknown 'ghost' dropped
    assert spec.prompt.startswith("ภาพ") and spec.image_prompt.startswith("A Shiba Inu")


def test_parser_agent_uses_llm(parsed_query):
    llm = FakeLLM({"ParsedQuery": parsed_query})
    spec = QueryParserAgent(llm).run("A Shiba Inu sitting in front of Wat Arun at sunset")
    assert [e.id for e in spec.entities] == ["wat_arun", "shiba_inu"]
    assert spec.scene.requirements == ["at sunset"]
    assert llm.calls == 1
