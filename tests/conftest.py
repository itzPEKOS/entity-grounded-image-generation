"""Fakes for every external dependency, so the pipeline logic is tested without network,
API keys or GPUs."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egig.backends import estimate_tokens  # noqa: E402
from egig.config import Settings  # noqa: E402
from egig.schemas import (Category, ParsedEntity, ParsedQuery, Relation, SceneSpec, VisualProfile,  # noqa: E402
                          VlmVerdict, VlmVerdicts)
from egig.vision import RawDetection  # noqa: E402
from egig.wiki import CommonsImage, WikiPage  # noqa: E402


class FakeLLM:
    """Answers by schema name. A value can be a model, a list (used in order) or a function of the parts."""

    def __init__(self, responses: Dict[str, object]):
        self.responses = responses
        self.calls = 0
        self.log: List[Tuple[str, list, bool]] = []

    def generate_json(self, schema, parts, *, system, vision=False):
        self.calls += 1
        self.log.append((schema.__name__, list(parts), vision))
        answer = self.responses[schema.__name__]
        if isinstance(answer, list):
            answer = answer.pop(0)
        if callable(answer) and not hasattr(answer, "model_dump"):
            answer = answer(parts)
        return answer


class FakeVision:
    def __init__(self, boxes: Dict[str, List[Tuple[float, List[float]]]]):
        self.boxes = boxes
        self.device = "cpu"

    def detect(self, image, label, box_threshold=None):
        return [RawDetection(label=label, score=s, box=b) for s, b in self.boxes.get(label, [])]

    def image_embeddings(self, images):
        vecs = np.array([np.asarray(im.convert("RGB").resize((8, 8)), dtype=float).mean(axis=(0, 1)) + 1 for im in images])
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def text_embeddings(self, texts):
        return np.ones((len(texts), 3)) / np.sqrt(3)


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.jobs = []

    def count_tokens(self, text):
        return estimate_tokens(text)

    def generate(self, job):
        self.jobs.append(job)
        return Image.new("RGB", (job.width, job.height), (90, 140, 200))


def striped_image(seed: str, size=(640, 480)) -> Image.Image:
    """Deterministic picture with strong edges (so Canny finds something)."""
    h = int(hashlib.md5(seed.encode()).hexdigest()[:6], 16)
    img = Image.new("RGB", size, ((h >> 16) & 255, (h >> 8) & 255, h & 255))
    draw = ImageDraw.Draw(img)
    for x in range(0, size[0], 40):
        draw.rectangle([x, 60, x + 18, size[1] - 60], fill=(255, 255, 255))
    return img


class FakeWiki:
    def __init__(self):
        self.downloads: List[str] = []

    def find_page(self, query):
        return WikiPage(title=query, url=f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                        extract=f"{query} is a well-known thing.", image_url=f"https://img.test/{query}/lead.jpg")

    def commons_images(self, query, limit=8):
        return [CommonsImage(title=f"File:{query} {i}.jpg", url=f"https://img.test/{query}/c{i}.jpg",
                             page_url="", width=800, height=600) for i in range(2)]

    def download_image(self, url, dest, min_side=256):
        self.downloads.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        striped_image(url).save(dest, format="JPEG")
        return dest


# ------------------------------------------------------------------------- data
@pytest.fixture
def parsed_query() -> ParsedQuery:
    return ParsedQuery(
        english_prompt="A Shiba Inu sitting in front of Wat Arun at sunset",
        scene=SceneSpec(setting="riverside in Bangkok", style="photorealistic", requirements=["at sunset"]),
        entities=[
            ParsedEntity(id="wat_arun", name="Wat Arun", category=Category.landmark, search_query="Wat Arun",
                         detection_label="temple", count=1, attributes=[], box=[0.12, 0.03, 0.88, 0.72],
                         layer=0, needs_reference=True, rigid=True),
            ParsedEntity(id="shiba_inu", name="Shiba Inu", category=Category.animal, search_query="Shiba Inu",
                         detection_label="dog", count=1, attributes=["sitting"], box=[0.32, 0.48, 0.68, 0.98],
                         layer=2, needs_reference=True, rigid=False),
        ],
        relations=[Relation(subject="shiba_inu", predicate="sitting in front of", object="wat_arun")],
    )


def profile_for(parts) -> VisualProfile:
    first = str(parts[0])
    if "Shiba" in first:
        return VisualProfile(summary="Small spitz dog.", key_features=["upright triangular ears", "curled tail",
                             "cream urajiro markings"], negative_traits=["floppy ears"],
                             confusables=["Akita Inu", "fox"], prompt_keywords="red Shiba Inu, curled tail")
    return VisualProfile(summary="Khmer-style prang temple.", key_features=["tall central prang", "porcelain mosaics",
                         "four smaller prangs", "steep stairs"], negative_traits=["golden chedi"],
                         confusables=["Wat Pho"], prompt_keywords="Wat Arun prang, porcelain mosaic")


def vlm_answer(fail: Tuple[str, ...] = ()) -> Callable:
    """VLM fake: passes every check id it is asked about, except those in `fail`."""
    def answer(parts):
        text = str(parts[-1])
        ids = [line.split("check_id=")[1].split(" |")[0] for line in text.splitlines() if "check_id=" in line]
        return VlmVerdicts(verdicts=[
            VlmVerdict(check_id=i, passed=i not in fail, confidence=0.9,
                       evidence="looks right" if i not in fail else "not as requested",
                       fix_hint="" if i not in fail else f"fix {i}")
            for i in ids])
    return answer


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(output_dir=str(tmp_path / "outputs"), image_size=256, max_attempts=3, llm_min_interval_s=0)


@pytest.fixture
def plan_matching_vision() -> FakeVision:
    """Detections that sit exactly where the parser planned the entities."""
    return FakeVision({"temple": [(0.8, [0.15, 0.05, 0.85, 0.7])], "dog": [(0.75, [0.35, 0.5, 0.65, 0.95])]})
