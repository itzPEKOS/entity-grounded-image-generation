"""Compare the system's verdicts with human labels -> eval_report.md

    python scripts/eval_report.py outputs/eval_dev --labels outputs/eval_dev/labels.csv
    python scripts/eval_report.py outputs/eval_dev --labels a.csv b.csv --system outputs/eval_dev_v2

--system: read the verdicts from a reverify.py output folder instead of the original runs
          (same images and labels, new verifier) -> compare two verifier versions fairly.

Four boxes per image (system verdict x human label):
    correct PASS | false PASS (verifier let a wrong image through: the costly one)
    false FAIL (verifier too strict) | correct FAIL (image wrong and caught: a generator problem)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from egig.evaluation import compare, find_runs, load_run, read_labels  # noqa: E402


def pct(a: int, b: int) -> str:
    return f"{a}/{b} = {100 * a / b:.0f}%" if b else f"{a}/0"


def render(rows, commits, title: str) -> str:
    atts = [a for r in rows for a in r.attempts]
    box = Counter(a.outcome for a in atts)
    judged = [a for a in atts if a.outcome not in ("unlabelled", "not verified")]
    sys_pass = [a for a in judged if a.system_pass]
    sys_fail = [a for a in judged if not a.system_pass]
    finals = [r for r in rows if r.final.label is not None]
    L = [f"# {title}", ""]
    if len(commits) > 1:
        L += [f"> WARNING: runs come from different commits {sorted(commits)}: they are not comparable.", ""]
    elif commits:
        L += [f"Code version: `{next(iter(commits))}`", ""]
    L += ["## Headline numbers", "",
          f"- Prompts: {len(rows)}, images: {len(atts)}, labelled: {len(judged)}"
          f", unlabelled: {box['unlabelled']}, not verified (Gemini down): {box['not verified']}",
          f"- **Verifier agrees with people**: {pct(box['correct PASS'] + box['correct FAIL'], len(judged))}",
          f"- **False PASS** (system PASS, person says wrong) among system PASSes: {pct(box['false PASS'], len(sys_pass))}",
          f"- **False FAIL** (system FAIL, person says OK) among system FAILs: {pct(box['false FAIL'], len(sys_fail))}",
          f"- **Final image is really correct** (what the user gets): {pct(sum(r.final.label.ok for r in finals), len(finals))}",
          f"- System says PASS for the final image: {pct(sum(r.status == 'PASS' for r in rows), len(rows))}",
          f"- First attempt correct (people) vs final correct: "
          f"{pct(sum(1 for r in rows if r.attempts[0].label and r.attempts[0].label.ok), sum(1 for r in rows if r.attempts[0].label))}"
          f" -> {pct(sum(r.final.label.ok for r in finals), len(finals))} (does the repair loop help?)",
          "", "## Four boxes (every labelled image)", "",
          "| | person: OK | person: wrong |", "|---|---|---|",
          f"| **system PASS** | correct PASS: {box['correct PASS']} | false PASS: {box['false PASS']} |",
          f"| **system FAIL** | false FAIL: {box['false FAIL']} | correct FAIL: {box['correct FAIL']} |", ""]

    def tag_table(title, items):
        c = Counter(t for a in items for t in (a.label.tags or ["(no tag)"]))
        if not items:
            return []
        out = [f"### {title} ({len(items)} images)", "", "| cause | images |", "|---|---|"]
        out += [f"| {t} | {n} |" for t, n in c.most_common()]
        return out + [""]

    L += ["## Causes (group, then fix the biggest group first)", ""]
    L += tag_table("Wrong images the verifier MISSED (false PASS) -> fix Agent 4",
                   [a for a in atts if a.outcome == "false PASS"])
    ff = [a for a in atts if a.outcome == "false FAIL"]
    if ff:
        c = Counter(k.split(":")[0] for a in ff for k in a.failed_checks)
        L += [f"### Good images the verifier REJECTED (false FAIL) -> fix Agent 4 ({len(ff)} images)", "",
              "Which kind of check failed them:", "", "| check kind | times |", "|---|---|"]
        L += [f"| {k} | {n} |" for k, n in c.most_common()] + [""]
    L += tag_table("Wrong images (caught or not) -> what Agents 1-3 get wrong",
                   [a for a in atts if a.label is not None and not a.label.ok])

    L += ["## Per prompt", "",
          "| id | request | system | final | person (final) | every attempt |", "|---|---|---|---|---|---|"]
    for r in rows:
        person = "-" if r.final.label is None else ("OK" if r.final.label.ok else "wrong: " + ", ".join(r.final.label.tags))
        each = "<br>".join(f"{a.attempt}: {a.outcome}" + (f" ({'; '.join(a.failed_checks[:3])})"
                                                          if a.outcome == "false FAIL" else "") for a in r.attempts)
        flag = " (needs GPU rerun)" if r.incomplete else ""
        L.append(f"| {r.run_id} | {r.prompt[:70]} | {r.status}{flag} | {r.final_attempt} | {person} | {each} |")
    notes = [(a.run_id, a.attempt, a.label.note) for a in atts if a.label and a.label.note]
    if notes:
        L += ["", "## Notes from people", ""] + [f"- {i} attempt {n}: {t}" for i, n, t in notes]
    if any(r.incomplete for r in rows):
        L += ["", "_needs GPU rerun: with these verdicts the loop would have made another attempt that was never "
                  "generated, so the final result could differ on a real run._"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, help="folder with the run folders (images), e.g. outputs/eval_dev")
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--system", type=Path, default=None, help="reverify.py output folder (default: the runs)")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    labels = read_labels(args.labels)
    runs = []
    for d in find_runs(args.runs):
        reports_dir = args.system / d.name if args.system else None
        if reports_dir is not None and not reports_dir.exists():
            print(f"[eval] {d.name}: not in {args.system}, skipped")
            continue
        runs.append(load_run(d, reports_dir))
    commits = set()
    for s in args.runs.glob("summary*.csv"):
        with open(s, encoding="utf-8-sig", newline="") as f:
            commits |= {r["commit"] for r in csv.DictReader(f) if r.get("commit")}
    rows = compare(runs, labels, args.max_attempts)
    title = f"Evaluation: {args.runs.name}" + (f" (verdicts from {args.system.name})" if args.system else "")
    text = render(rows, commits, title)
    out = args.out or (args.system or args.runs) / "eval_report.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
