"""Feedback loop policy: how Agent 3 changes its knobs after Agent 4 reports failed checks.

Every change is written to `notes` with the reason, so the report shows *why* attempt N+1
differs from attempt N.

Knob cheat-sheet
- ip_scale[e]   : how strongly entity e copies its reference photo (IP-Adapter).
                  Higher = looks more like the real thing, but also copies the photo's pose,
                  so it can fight the user's request ("sitting") and the scene ("sunset").
- control_scale : how strongly rigid entities follow the reference edges (ControlNet).
- emphasis[e]   : moves entity e earlier in the prompt (earlier words weigh more in SDXL).
- extra_prompt / extra_negative : phrases added to the prompt / negative prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .schemas import AttemptReport, CheckKind, CheckResult, ConstraintSpec, GenParams, Severity


@dataclass
class Limits:
    ip_min: float = 0.3
    ip_max: float = 0.9
    control_min: float = 0.2
    control_max: float = 0.9
    max_extra_prompt: int = 6
    max_extra_negative: int = 6


def _clip(v: float, lo: float, hi: float) -> float:
    return round(min(max(v, lo), hi), 2)


def _add(items: List[str], phrase: str, limit: int) -> None:
    """Newest first, no duplicates, bounded length."""
    phrase = phrase.strip().rstrip(".")
    if not phrase:
        return
    if phrase in items:
        items.remove(phrase)
    items.insert(0, phrase)
    del items[limit:]


def repair(params: GenParams, report: AttemptReport, spec: ConstraintSpec, limits: Limits = Limits()) -> GenParams:
    new = params.model_copy(deep=True)
    new.notes = []
    new.seed = params.seed + 1
    new.notes.append(f"seed {params.seed} -> {new.seed} (new random start)")

    def bump_ip(entity_id: str, delta: float, reason: str) -> None:
        if entity_id not in new.ip_scale:
            return
        old = new.ip_scale[entity_id]
        new.ip_scale[entity_id] = _clip(old + delta, limits.ip_min, limits.ip_max)
        if new.ip_scale[entity_id] != old:
            new.notes.append(f"ip_scale[{entity_id}] {old:.2f} -> {new.ip_scale[entity_id]:.2f} ({reason})")

    def bump_control(delta: float, reason: str) -> None:
        old = new.control_scale
        new.control_scale = _clip(old + delta, limits.control_min, limits.control_max)
        if new.control_scale != old:
            new.notes.append(f"control_scale {old:.2f} -> {new.control_scale:.2f} ({reason})")

    def emphasize(entity_id: str, reason: str) -> None:
        new.emphasis[entity_id] = new.emphasis.get(entity_id, 0) + 1
        new.notes.append(f"emphasis[{entity_id}] -> {new.emphasis[entity_id]} ({reason})")

    def prompt(phrase: str, reason: str) -> None:
        _add(new.extra_prompt, phrase, limits.max_extra_prompt)
        new.notes.append(f"prompt += '{phrase}' ({reason})")

    failed: List[CheckResult] = sorted(report.failed(), key=lambda r: r.severity != Severity.critical)
    for r in failed:
        e = spec.entity(r.entity_id)
        why = f"{r.check_id} failed"
        if r.kind == CheckKind.presence and e:
            emphasize(e.id, why)
            bump_ip(e.id, +0.10, why)
            prompt(f"a clearly visible {e.name}", why)
        elif r.kind == CheckKind.count and e:
            emphasize(e.id, why)
            prompt(f"exactly {e.count} {e.detection_label}s", why)
        elif r.kind == CheckKind.position and e:
            if e.rigid:
                bump_control(+0.10, why + ": follow the planned structure more")
            emphasize(e.id, why)
        elif r.kind == CheckKind.identity and e:
            bump_ip(e.id, +0.15, why + ": look more like the reference photo")
            for neg in (e.profile.negative_traits if e.profile else []):
                _add(new.extra_negative, neg, limits.max_extra_negative)
        elif r.kind == CheckKind.feature and e:
            bump_ip(e.id, +0.05, why)
            feature = r.description.split(":", 1)[-1].strip()
            prompt(f"{e.name} with {feature}", why)
        elif r.kind == CheckKind.attribute and e:
            bump_ip(e.id, -0.15, why + ": a strong reference copies the photo's pose and overrides the request")
            emphasize(e.id, why)
            attribute = r.description.split(":", 1)[-1].strip()
            prompt(f"{e.name} {attribute}", why)
        elif r.kind == CheckKind.scene:
            for eid in list(new.ip_scale):
                bump_ip(eid, -0.05, why + ": reference photos carry their own lighting")
            prompt(r.description.split(":", 1)[-1].strip(), why)
        elif r.kind == CheckKind.relation:
            prompt(r.description.split(":", 1)[-1].strip(), why)
        # r.fix_hint (Gemini's free-text advice) is NOT pasted into the prompt: it is a full sentence
        # that got cut mid-way by the 75-token limit and repeated what the rules above already add.
        # It stays in the report for people to read.
    return new
