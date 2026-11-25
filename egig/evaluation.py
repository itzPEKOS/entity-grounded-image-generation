"""Evaluation helpers: run a fixed prompt set in one go, re-check saved images with the current
verifier, and compare the system's verdicts with human labels.

    eval/prompts.csv            the fixed prompt set (id, split, part, category, prompt)
    run_batch(...)              Kaggle: run every prompt, survive Gemini errors, write summary_<part>.csv
    scripts/reverify.py         laptop: run the verifier again on saved images (no SDXL, no GPU)
    scripts/make_review.py      laptop: review.html for people to label each image
    scripts/eval_report.py      laptop: system vs human -> false PASS / false FAIL / causes
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .agents.verifier import VerificationAgent, build_checks, score_tuple
from .schemas import AttemptReport, Check, CheckResult, ConstraintSpec, Method

PROMPT_FIELDS = ("id", "split", "part", "category", "prompt")
SUMMARY_FIELDS = ("id", "part", "split", "category", "prompt", "status", "final_attempt", "attempts",
                  "critical", "major_rate", "gemini_calls", "seconds", "commit", "error")
DONE = ("PASS", "FAIL")  # statuses that need no rerun; NOT_VERIFIED and ERROR are run again


@dataclass
class EvalPrompt:
    id: str
    prompt: str
    split: str = "dev"
    part: str = ""
    category: str = ""


def load_prompts(path: Path, split: Optional[str] = None, part: Optional[str] = None,
                 only: Optional[Iterable[str]] = None) -> List[EvalPrompt]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [EvalPrompt(**{k: (r.get(k) or "").strip() for k in PROMPT_FIELDS}) for r in csv.DictReader(f)]
    ids = [r.id for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate prompt ids in {path}: {dupes}")
    only_set = {o.strip() for o in only} if only else None
    return [r for r in rows
            if (not split or r.split == split) and (not part or r.part == part)
            and (only_set is None or r.id in only_set)]


# ------------------------------------------------------------------ batch run (Kaggle)
def status_of(reports: Sequence[AttemptReport], best_index: int) -> str:
    if reports[best_index].verdict.passed:
        return "PASS"
    if reports[-1].vlm_error:  # the loop stopped because Gemini could not check the image
        return "NOT_VERIFIED"
    return "FAIL"


def read_summary(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["id"]: dict(r) for r in csv.DictReader(f)}


def write_summary(path: Path, rows: Dict[str, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in sorted(rows.values(), key=lambda r: r["id"]):
            w.writerow(row)
    tmp.replace(path)  # never leave a half-written summary behind


def _free_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_batch(orch, prompts: Sequence[EvalPrompt], subdir: str = "eval", summary_name: str = "summary.csv",
              commit: str = "", retry_wait_s: float = 120, outage_wait_s: float = 600,
              max_consecutive_errors: int = 3, sleep: Callable[[float], None] = time.sleep,
              rerun_not_verified: bool = True) -> Dict[str, Dict[str, str]]:
    """Run every prompt; one failing prompt never stops the batch.

    - each prompt is tried twice: an exception or a NOT_VERIFIED result (Gemini down) waits
      `retry_wait_s` and runs it again; a second failure is recorded and the batch moves on
    - `rerun_not_verified=False` (generator-only runs, max_attempts=1): the image is what is being
      tested and it is verified later on the laptop, so NOT_VERIFIED counts as done, not as a failure
    - `max_consecutive_errors` prompts in a row failing means Gemini is probably down: wait `outage_wait_s`
    - summary.csv is rewritten after every prompt, so a crash or a stopped session keeps what is done
    - running the same cell again skips prompts whose summary row is already PASS or FAIL
    """
    out_dir = Path(orch.s.output_dir) / subdir
    summary_path = out_dir / summary_name
    rows = read_summary(summary_path)
    consecutive_errors = 0
    done = DONE if rerun_not_verified else DONE + ("NOT_VERIFIED",)
    todo = [p for p in prompts if rows.get(p.id, {}).get("status") not in done]
    print(f"[batch] {len(prompts)} prompts, {len(prompts) - len(todo)} already done, {len(todo)} to run "
          f"-> {out_dir}")

    for n, p in enumerate(todo, 1):
        row: Dict[str, str] = {}
        for attempt in (1, 2):
            run_dir = out_dir / p.id
            if run_dir.exists():
                shutil.rmtree(run_dir)  # a half-finished run would mix old and new files
            started = time.monotonic()
            try:
                result = orch.run(p.prompt, run_id=f"{subdir}/{p.id}")
                best = result.reports[result.best_index]
                row = _row(p, commit, status=status_of(result.reports, result.best_index),
                           final_attempt=str(result.best_index + 1), attempts=str(len(result.reports)),
                           critical=f"{best.verdict.critical_passed}/{best.verdict.critical_total}",
                           major_rate=f"{best.verdict.major_rate:.2f}", gemini_calls=str(result.llm_calls),
                           seconds=f"{time.monotonic() - started:.0f}",
                           error=result.reports[-1].vlm_error[:300])
            except KeyboardInterrupt:
                raise
            except Exception as err:  # Gemini down for good, CUDA out of memory, network, ...
                traceback.print_exc()
                row = _row(p, commit, status="ERROR", seconds=f"{time.monotonic() - started:.0f}",
                           error=f"{type(err).__name__}: {err}"[:300])
                _free_gpu_memory()
            if row["status"] in done:
                break
            if attempt == 1:
                print(f"[batch] {p.id}: {row['status']} -> waiting {retry_wait_s:.0f}s, then trying once more")
                sleep(retry_wait_s)

        rows[p.id] = row
        write_summary(summary_path, rows)
        print(f"[batch] {n}/{len(todo)} {p.id}: {row['status']} ({row['seconds']}s)")

        consecutive_errors = consecutive_errors + 1 if row["status"] not in done else 0
        if consecutive_errors >= max_consecutive_errors and n < len(todo):
            print(f"[batch] {consecutive_errors} prompts in a row failed: Gemini is probably down, "
                  f"waiting {outage_wait_s:.0f}s")
            sleep(outage_wait_s)
            consecutive_errors = 0

    counts = {s: sum(1 for r in rows.values() if r["status"] == s) for s in ("PASS", "FAIL", "NOT_VERIFIED", "ERROR")}
    print(f"[batch] finished: {counts}")
    rerun = [r["id"] for r in rows.values() if r["status"] not in done]
    if rerun:
        print(f"[batch] run again later (same cell, or ONLY = {rerun})")
    return rows


def _row(p: EvalPrompt, commit: str, **values: str) -> Dict[str, str]:
    row = {k: "" for k in SUMMARY_FIELDS}
    row.update(id=p.id, part=p.part, split=p.split, category=p.category, prompt=p.prompt, commit=commit)
    row.update(values)
    return row


# ------------------------------------------------------------------ saved runs (laptop)
@dataclass
class SavedRun:
    id: str
    dir: Path
    prompt: str
    spec: ConstraintSpec
    reports: List[AttemptReport] = field(default_factory=list)  # attempt 1, 2, ... in order

    def image(self, attempt: int) -> Path:
        return self.dir / f"attempt_{attempt}" / "image.png"


def find_runs(root: Path) -> List[Path]:
    """Every run folder under root (a folder holding 02_spec_grounded.json), sorted by name."""
    return sorted(p.parent for p in Path(root).rglob("02_spec_grounded.json"))


def localize_refs(spec: ConstraintSpec, run_dir: Path) -> ConstraintSpec:
    """Reference paths are saved as Kaggle paths (/kaggle/working/outputs/<run>/refs/...);
    point them at the refs/ folder next to the spec instead."""
    spec = spec.model_copy(deep=True)
    for e in spec.entities:
        for r in e.references:
            for attr in ("path", "crop_path"):
                value = getattr(r, attr).replace("\\", "/")
                if "/refs/" in value:
                    setattr(r, attr, str(run_dir / "refs" / value.split("/refs/", 1)[1]))
    return spec


def load_run(run_dir: Path, reports_dir: Optional[Path] = None) -> SavedRun:
    """Load a run. reports_dir (default: run_dir) is where attempt_<n>/report.json are read from,
    so the same loader reads original reports and re-verified ones."""
    run_dir = Path(run_dir)
    spec = ConstraintSpec.model_validate_json((run_dir / "02_spec_grounded.json").read_text(encoding="utf-8"))
    spec = localize_refs(spec, run_dir)
    run = SavedRun(id=run_dir.name, dir=run_dir, prompt=spec.prompt, spec=spec)
    src = Path(reports_dir) if reports_dir else run_dir
    n = 1
    while (src / f"attempt_{n}" / "report.json").exists():
        run.reports.append(AttemptReport.model_validate_json(
            (src / f"attempt_{n}" / "report.json").read_text(encoding="utf-8")))
        n += 1
    return run


def image_sig(path: Path) -> str:
    """Short fingerprint of an image file: a label belongs to these exact pixels. When a prompt is
    run again, its attempt_1/image.png is a new picture and the old label must not carry over."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:10]


def attempt_numbers(run_dir: Path) -> List[int]:
    n, out = 1, []
    while (Path(run_dir) / f"attempt_{n}" / "image.png").exists():
        out.append(n)
        n += 1
    return out


def replay_loop(reports: Sequence[AttemptReport], max_attempts: int) -> Dict[str, object]:
    """What the pipeline's loop would have returned with these verdicts: stop at the first PASS or
    at a Gemini outage, then pick the best attempt seen so far.

    `incomplete` = the loop would have wanted another attempt that does not exist on disk
    (e.g. attempt 1 used to PASS, now FAILs, and attempt 2 was never generated) -> needs a GPU rerun."""
    seen: List[AttemptReport] = []
    for r in reports:
        seen.append(r)
        if r.verdict.passed or r.vlm_error:
            break
    best = max(range(len(seen)), key=lambda i: score_tuple(seen[i]))
    last = seen[-1]
    incomplete = not last.verdict.passed and not last.vlm_error and len(seen) < max_attempts
    return {"best_index": best, "status": status_of(seen, best), "used": len(seen), "incomplete": incomplete}


# ------------------------------------------------------------------ re-verify saved images (laptop)
class ReplayVerifier(VerificationAgent):
    """The current verifier, with an option to reuse Gemini's saved answers.

    reuse_vlm=True: Gemini checks and counts come from the old report.json (no API calls, no randomness),
    only the detector side and the decision rules run again -> right for tuning thresholds / count rules.
    reuse_vlm=False: Gemini is asked again -> right after changing the Gemini prompt or the checks."""

    def __init__(self, llm, vision, settings, reuse_vlm: bool = False):
        super().__init__(llm, vision, settings)
        self.reuse_vlm = reuse_vlm
        self.saved: Optional[AttemptReport] = None
        self.reused = 0

    def _ask_vlm(self, spec, checks, image, crops):
        old = self.saved
        if self.reuse_vlm and old is not None and not old.vlm_error:
            by_id: Dict[str, CheckResult] = {r.check_id: r for r in old.results if r.method == Method.vlm}
            if all(c.id in by_id for c in checks):
                self.reused += 1
                return {c.id: by_id[c.id] for c in checks}, dict(old.vlm_counts)
        return super()._ask_vlm(spec, checks, image, crops)


def reverify_run(run_dir: Path, out_dir: Path, verifier: ReplayVerifier, saved_checks: bool = False,
                 resume: bool = False) -> SavedRun:
    """Check every saved attempt image of one run again; write out_dir/attempt_<n>/report.json.

    resume=True keeps a report already in out_dir when Gemini answered it (no vlm_error) and it is
    newer than the image, so a run stopped by a Gemini outage only redoes what is missing."""
    old = load_run(run_dir)
    if saved_checks:
        raw = json.loads((Path(run_dir) / "03_checks.json").read_text(encoding="utf-8"))
        checks = [Check.model_validate(c) for c in raw]
    else:
        checks = build_checks(old.spec)  # the checks the CURRENT code would build
    new = SavedRun(id=old.id, dir=old.dir, prompt=old.prompt, spec=old.spec)
    for n in attempt_numbers(run_dir):
        attempt_dir = Path(out_dir) / f"attempt_{n}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        done = attempt_dir / "report.json"
        if resume and done.exists() and done.stat().st_mtime > old.image(n).stat().st_mtime:
            kept = AttemptReport.model_validate_json(done.read_text(encoding="utf-8"))
            if not kept.vlm_error:
                new.reports.append(kept)
                continue
        verifier.saved = old.reports[n - 1] if n <= len(old.reports) else None
        report = verifier.run(old.spec, checks, old.image(n), n, attempt_dir)
        if verifier.saved is not None:
            report.repair_notes = list(verifier.saved.repair_notes)
        (attempt_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        new.reports.append(report)
    verifier.saved = None
    return new


# ------------------------------------------------------------------ human labels vs system
TAGS = {  # what a person can say is wrong with an image
    "identity": "wrong breed / landmark / object",
    "count": "wrong number of things",
    "missing": "an entity is missing",
    "merged": "animals merged / deformed body",
    "frame": "picture frame, poster, panel or collage that was not asked for",
    "attribute": "wrong pose, action, colour or size",
    "position": "wrong left/right/top/bottom placement",
    "relation": "relation between entities is wrong",
    "scene": "wrong background or scene",
    "quality": "broken, blurry or unrealistic image",
    "other": "anything else",
}


@dataclass
class Label:
    run_id: str
    attempt: int
    ok: bool
    tags: List[str] = field(default_factory=list)
    note: str = ""
    labeler: str = ""
    image: str = ""  # image_sig() of the picture that was judged ("" in old files)


def read_labels(paths: Iterable[Path]) -> Dict[tuple, Label]:
    """labels.csv files from review.html: run_id, attempt, human (ok|wrong), tags (a;b), note, labeler, image.
    Several files can be merged; a later file wins for the same image."""
    labels: Dict[tuple, Label] = {}
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                human = (r.get("human") or "").strip().lower()
                if human not in ("ok", "wrong"):
                    continue  # not labelled yet
                tags = [t.strip() for t in (r.get("tags") or "").split(";") if t.strip()]
                lab = Label(r["run_id"].strip(), int(r["attempt"]), human == "ok", tags,
                            (r.get("note") or "").strip(), (r.get("labeler") or "").strip(),
                            (r.get("image") or "").strip())
                labels[(lab.run_id, lab.attempt)] = lab
    return labels


def outcome(system_pass: bool, human_ok: bool) -> str:
    """The four boxes: the verifier is right on the diagonal; off it, a false PASS (wrong image let
    through) is the costly one."""
    if system_pass and human_ok:
        return "correct PASS"
    if system_pass:
        return "false PASS"
    if human_ok:
        return "false FAIL"
    return "correct FAIL"


@dataclass
class AttemptRow:
    run_id: str
    attempt: int
    system_pass: bool
    not_verified: bool
    label: Optional[Label]
    failed_checks: List[str]  # "kind:check_id" of checks the system failed

    @property
    def outcome(self) -> str:
        if self.not_verified:
            return "not verified"
        if self.label is None:
            return "unlabelled"
        return outcome(self.system_pass, self.label.ok)


@dataclass
class RunRow:
    run_id: str
    prompt: str
    status: str  # system's final status after replaying the loop
    final_attempt: int
    incomplete: bool
    attempts: List[AttemptRow]

    @property
    def final(self) -> AttemptRow:
        return self.attempts[self.final_attempt - 1]


def compare(runs: Sequence[SavedRun], labels: Dict[tuple, Label], max_attempts: int) -> List[RunRow]:
    rows: List[RunRow] = []
    for run in runs:
        if not run.reports:
            continue
        loop = replay_loop(run.reports, max_attempts)
        attempts = []
        for n, r in enumerate(run.reports, 1):
            label = labels.get((run.id, n))
            if label is not None and label.image and run.image(n).exists() and label.image != image_sig(run.image(n)):
                label = None  # judged a different picture (the prompt was run again): not valid here
            attempts.append(AttemptRow(run.id, n, r.verdict.passed, bool(r.vlm_error), label,
                                       [f"{c.kind.value}:{c.check_id}" for c in r.failed()]))
        rows.append(RunRow(run.id, run.prompt, str(loop["status"]), int(loop["best_index"]) + 1,
                           bool(loop["incomplete"]), attempts))
    return rows
