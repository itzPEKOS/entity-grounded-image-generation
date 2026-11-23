"""Agent 2 - Entity Retriever: for every entity that needs it, fetch real reference photos
(Wikipedia lead image + Wikimedia Commons), keep the ones that really show the entity, crop
them to the entity, and write a visual profile (what it really looks like)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PIL import Image

from ..config import Settings
from ..geometry import expand, to_pixels, union_box
from ..llm import LLM
from ..schemas import ConstraintSpec, EntitySpec, ReferenceImage, VisualProfile
from ..vision import VisionTools, cosine
from ..wiki import WikiClient, WikiPage

SYSTEM = """You are the Entity Retriever agent of an entity-grounded text-to-image system.
Describe what the given real-world entity actually looks like, grounded in the Wikipedia text
and the reference photo(s). Only list features that are visible in a normal photo.
- key_features: 3-5 short, checkable visual features (shape, colours, materials, markings, proportions).
- negative_traits: up to 3 short noun phrases for common WRONG depictions (e.g. "floppy ears", "golden chedi").
  Never contradict what the user asked for.
- confusables: up to 4 things that look similar and are often confused with it.
- prompt_keywords: max 15 words, comma separated, to describe it to an image model."""


class EntityRetrieverAgent:
    name = "EntityRetriever"

    def __init__(self, llm: LLM, wiki: WikiClient, vision: Optional[VisionTools], settings: Settings):
        self.llm = llm
        self.wiki = wiki
        self.vision = vision
        self.s = settings

    def run(self, spec: ConstraintSpec, refs_dir: Path) -> ConstraintSpec:
        grounded = [self._ground(e, spec, refs_dir / e.id) for e in spec.entities]
        return spec.model_copy(update={"entities": grounded})

    # ----------------------------------------------------------------- per entity
    def _ground(self, e: EntitySpec, spec: ConstraintSpec, out_dir: Path) -> EntitySpec:
        if not e.needs_reference:
            print(f"[retriever] {e.id}: generic entity, no lookup needed")
            return e

        page = self.wiki.find_page(e.search_query or e.name)
        candidates = []
        if page and page.image_url:
            candidates.append((page.image_url, "wikipedia-lead"))
        for img in self.wiki.commons_images(e.search_query or e.name, limit=self.s.commons_candidates):
            if all(img.url != url for url, _ in candidates):
                candidates.append((img.url, "commons"))

        downloaded: List[ReferenceImage] = []
        for k, (url, source) in enumerate(candidates):
            path = self.wiki.download_image(url, out_dir / f"cand_{k:02d}.jpg")
            if path is not None:
                downloaded.append(ReferenceImage(path=str(path), crop_path=str(path), url=url, source=source))

        refs = self._curate(e, downloaded, out_dir)
        print(f"[retriever] {e.id}: page={page.title if page else None!r}, "
              f"{len(downloaded)} photos downloaded, {len(refs)} kept")
        profile = self._profile(e, spec, page, refs)
        sources = ([page.url] if page and page.url else []) + [r.url for r in refs]
        return e.model_copy(update={"references": refs, "profile": profile, "source_urls": sources})

    def _curate(self, e: EntitySpec, cands: List[ReferenceImage], out_dir: Path) -> List[ReferenceImage]:
        """Keep photos where the detector finds the entity, rank them by CLIP similarity to a
        text description, crop each to the entity.
        - rigid (temple, bridge): crop = union of all detected parts, so every tower stays in
        - non-rigid (dog, person): crop = best single box; prefer full-body shots"""
        if not cands:
            return []
        if self.vision is None:  # no torch: trust the order (Wikipedia lead image first)
            return cands[: self.s.refs_per_entity]

        images = [Image.open(c.path).convert("RGB") for c in cands]
        description = f"a photo of {e.name}" if e.rigid else f"a full-body photo of a {e.name}"
        text_emb = self.vision.text_embeddings([description])[0]
        img_embs = self.vision.image_embeddings(images)
        scored = []
        for c, img, emb in zip(cands, images, img_embs):
            dets = self.vision.detect(img, e.detection_label, box_threshold=self.s.ref_box_threshold)
            c.det_score = round(max((d.score for d in dets), default=0.0), 3)
            c.clip_score = round(cosine(emb, text_emb), 4)
            if dets:
                box = union_box([d.box for d in dets]) if e.rigid else max(dets, key=lambda d: d.score).box
                c.crop_box = [round(v, 4) for v in expand(box, 0.04)]
            scored.append((c, img))

        kept = [(c, img) for c, img in scored if c.crop_box is not None]
        if not kept:
            print(f"[retriever] {e.id}: detector found no '{e.detection_label}' in any photo; using uncropped photos")
            kept = scored
        # CLIP scores of good photos are close together (e.g. 0.352 vs 0.353), so a small bonus lets
        # the Wikipedia lead photo (usually the canonical view) win ties.
        kept.sort(key=lambda ci: (ci[0].clip_score or 0.0) + (0.02 if ci[0].source == "wikipedia-lead" else 0.0),
                  reverse=True)

        refs = []
        for c, img in kept[: self.s.refs_per_entity]:
            if c.crop_box is not None:
                crop = img.crop(to_pixels(c.crop_box, img.width, img.height))
                crop_path = out_dir / (Path(c.path).stem + "_crop.jpg")
                crop.save(crop_path, quality=92)
                c.crop_path = str(crop_path)
            refs.append(c)
        return refs

    def _profile(self, e: EntitySpec, spec: ConstraintSpec, page: Optional[WikiPage],
                 refs: List[ReferenceImage]) -> VisualProfile:
        parts: list = [
            f"Entity: {e.name} (category: {e.category.value})",
            f"The user wants: {spec.prompt!r}. Requested attributes for this entity: {e.attributes or 'none'}",
            f"Wikipedia ({page.title if page else 'not found'}): {(page.extract if page else '')[:1500]}",
        ]
        if refs:
            parts += ["Reference photo:", Image.open(refs[0].crop_path)]
        p = self.llm.generate_json(VisualProfile, parts, system=SYSTEM, vision=bool(refs))
        return VisualProfile(
            summary=p.summary.strip(),
            key_features=[f.strip() for f in p.key_features if f.strip()][:5],
            negative_traits=[n.strip() for n in p.negative_traits if n.strip()][:3],
            confusables=[c.strip() for c in p.confusables if c.strip()][:4],
            prompt_keywords=" ".join(p.prompt_keywords.split()[:15]),
        )
