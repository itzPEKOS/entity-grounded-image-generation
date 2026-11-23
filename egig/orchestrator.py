"""Orchestrator: runs the four agents and the feedback loop, and saves every message.

    user prompt
      -> Agent 1 QueryParser      -> constraint spec
      -> Agent 2 EntityRetriever  -> grounded spec (+ reference photos)
      -> loop (max_attempts):
           Agent 3 ImageGenerator -> image
           Agent 4 Verifier       -> report
           PASS -> stop; FAIL -> report goes back to Agent 3, which repairs its knobs
      -> best attempt + final report
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from .agents import EntityRetrieverAgent, ImageGeneratorAgent, QueryParserAgent, VerificationAgent
from .agents.verifier import build_checks, score_tuple
from .backends import Backend
from .config import Settings
from .llm import LLM, GeminiLLM
from .report import attempt_markdown, final_markdown
from .schemas import AttemptReport, ConstraintSpec, GenerationRecord
from .vision import VisionTools, cuda_available, default_vision_device, vision_available
from .wiki import WikiClient


class Trace:
    """Append-only log of messages between agents (trace.jsonl)."""

    def __init__(self, path: Path):
        self.path = path

    def log(self, sender: str, receiver: str, kind: str, summary: str, payload: Optional[BaseModel] = None) -> None:
        row = {"time": datetime.now().isoformat(timespec="seconds"), "from": sender, "to": receiver,
               "message": kind, "summary": summary}
        if payload is not None:
            row["payload"] = json.loads(payload.model_dump_json())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {sender} -> {receiver}: {kind} ({summary})")


@dataclass
class RunResult:
    run_dir: Path
    spec: ConstraintSpec
    reports: List[AttemptReport]
    records: List[GenerationRecord]
    best_index: int
    final_image: Path

    @property
    def passed(self) -> bool:
        return self.reports[self.best_index].verdict.passed


class Orchestrator:
    def __init__(self, settings: Settings, llm: LLM, backend: Backend, vision: Optional[VisionTools],
                 wiki: Optional[WikiClient] = None):
        self.s = settings
        self.llm = llm
        self.parser = QueryParserAgent(llm)
        self.retriever = EntityRetrieverAgent(
            llm, wiki or WikiClient(settings.wiki_contact, settings.wiki_lang), vision, settings)
        self.generator = ImageGeneratorAgent(backend, settings)
        self.verifier = VerificationAgent(llm, vision, settings)

    def run(self, prompt: str, run_id: Optional[str] = None) -> RunResult:
        run_dir = Path(self.s.output_dir) / (run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
        run_dir.mkdir(parents=True, exist_ok=True)
        trace = Trace(run_dir / "trace.jsonl")
        started = time.monotonic()
        print(f"[run] {run_dir}")

        trace.log("user", "QueryParser", "prompt", prompt)
        spec = self.parser.run(prompt)
        _save(run_dir / "01_spec.json", spec)
        trace.log("QueryParser", "EntityRetriever", "constraint_spec",
                  f"{len(spec.entities)} entities: {[e.id for e in spec.entities]}", spec)

        spec = self.retriever.run(spec, run_dir / "refs")
        _save(run_dir / "02_spec_grounded.json", spec)
        checks = build_checks(spec)
        (run_dir / "03_checks.json").write_text(
            json.dumps([c.model_dump(mode="json") for c in checks], indent=2, ensure_ascii=False), encoding="utf-8")
        trace.log("EntityRetriever", "ImageGenerator", "grounded_spec",
                  ", ".join(f"{e.id}: {len(e.references)} refs" for e in spec.entities), spec)

        params = self.generator.initial_params(spec)
        reports: List[AttemptReport] = []
        records: List[GenerationRecord] = []
        for attempt in range(1, self.s.max_attempts + 1):
            attempt_dir = run_dir / f"attempt_{attempt}"
            record = self.generator.run(spec, params, attempt_dir, attempt)
            trace.log("ImageGenerator", "Verifier", "image", record.image_path, record)

            report = self.verifier.run(spec, checks, Path(record.image_path), attempt, attempt_dir)
            report.repair_notes = list(params.notes)
            _save(attempt_dir / "report.json", report)
            (attempt_dir / "report.md").write_text(attempt_markdown(report, record, spec), encoding="utf-8")
            reports.append(report)
            records.append(record)
            print(f"[verifier] attempt {attempt}: {report.verdict.explanation}")

            if report.verdict.passed:
                trace.log("Verifier", "Orchestrator", "PASS", report.verdict.explanation, report)
                break
            if report.vlm_error:  # no real feedback to repair from: keep what we have instead of crashing
                trace.log("Verifier", "Orchestrator", "STOP (Gemini unavailable)", report.vlm_error[:200], report)
                print("[run] Gemini is unavailable, so this attempt could not be checked; stopping and "
                      "keeping the best attempt so far")
                break
            if attempt == self.s.max_attempts:
                trace.log("Verifier", "Orchestrator", "FAIL (no attempts left)", report.verdict.explanation, report)
                break
            trace.log("Verifier", "ImageGenerator", "FAIL + feedback",
                      f"{len(report.failed())} failed: {[r.check_id for r in report.failed()]}", report)
            params = self.generator.repair(params, report, spec)

        best_index = max(range(len(reports)), key=lambda i: score_tuple(reports[i]))
        final_image = run_dir / "final.png"
        shutil.copyfile(records[best_index].image_path, final_image)
        (run_dir / "final_report.md").write_text(
            final_markdown(spec, reports, records, best_index, self.llm.calls), encoding="utf-8")
        print(f"[run] done in {time.monotonic() - started:.0f}s: "
              f"{'PASS' if reports[best_index].verdict.passed else 'FAIL'} (best = attempt {best_index + 1}), "
              f"Gemini calls: {self.llm.calls}")
        return RunResult(run_dir, spec, reports, records, best_index, final_image)


def _save(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


# ---------------------------------------------------------------------- factory
def make_backend(settings: Settings) -> Backend:
    from .backends.collage import CollageBackend

    choice = settings.generator
    if choice == "collage" or (choice == "auto" and not cuda_available()):
        if choice == "auto":
            print("[setup] no CUDA GPU -> using the collage stub instead of SDXL (fine for testing on a laptop)")
        return CollageBackend()
    from .backends.sdxl import SDXLBackend

    return SDXLBackend(settings, device="cuda:0")


def make_vision(settings: Settings) -> Optional[VisionTools]:
    if settings.detector == "none":
        return None
    if not vision_available():
        if settings.detector == "gdino":
            raise RuntimeError("DETECTOR=gdino needs torch + transformers (pip install -r requirements-vision.txt)")
        print("[setup] torch/transformers not installed -> detector checks will be skipped")
        return None
    return VisionTools(settings.gdino_model, settings.clip_model, device=default_vision_device(),
                       box_threshold=settings.box_threshold, text_threshold=settings.text_threshold)


def build_orchestrator(settings: Settings, llm: Optional[LLM] = None, backend: Optional[Backend] = None,
                       vision: Optional[VisionTools] = None, wiki: Optional[WikiClient] = None) -> Orchestrator:
    llm = llm or GeminiLLM(settings.gemini_api_key, settings.llm_model, settings.vlm_model,
                           settings.llm_min_interval_s, settings.llm_max_retries, settings.llm_temperature,
                           fallback_models=settings.llm_fallback_models)
    backend = backend or make_backend(settings)
    vision = vision if vision is not None else make_vision(settings)
    return Orchestrator(settings, llm, backend, vision, wiki)
