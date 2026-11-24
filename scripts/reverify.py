"""Run the CURRENT verifier (Agent 4) again on images saved by earlier runs. No SDXL, no GPU.

    python scripts/reverify.py outputs/eval_dev --out outputs/eval_dev_v2 [--reuse-vlm] [--env ../entity-grounded-gen/.env]

Use it after changing only the verifier (thresholds, count rules, check building, Gemini's verify prompt):
the images stay the same, so any change in the verdicts comes from your code, not from diffusion randomness.
After changing the parser (Agent 1), retriever (Agent 2) or generator (Agent 3), run the prompts again on Kaggle.

--reuse-vlm      reuse Gemini's saved answers; only the detector and the decision rules run again
                 (no Gemini calls, same answers every time: right for thresholds and count rules)
--saved-checks   use the checks saved in 03_checks.json instead of building them with the current code
--offline        load Grounding DINO / CLIP from the local cache only (much faster when the Hub is slow or blocked)

CPU speed: about 1 minute per image, so 15 prompts x 3 attempts is roughly 45-60 minutes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from egig.config import Settings  # noqa: E402
from egig.evaluation import ReplayVerifier, find_runs, load_run, replay_loop, reverify_run  # noqa: E402
from egig.llm import LLMError  # noqa: E402


class NoLLM:
    """Stands in when there is no API key: every Gemini check becomes NOT VERIFIED."""
    calls = 0

    def generate_json(self, schema, parts, *, system, vision=False):
        raise LLMError("no GEMINI_API_KEY (run with --reuse-vlm, or pass --env path/to/.env)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, help="folder holding the run folders (e.g. outputs/eval_dev)")
    ap.add_argument("--out", type=Path, required=True, help="where the new reports go (a new folder)")
    ap.add_argument("--reuse-vlm", action="store_true")
    ap.add_argument("--saved-checks", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated run ids")
    ap.add_argument("--env", type=Path, default=None, help=".env file with GEMINI_API_KEY (default: ./.env)")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.offline:
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"  # must be set before transformers is imported

    try:
        from dotenv import load_dotenv

        load_dotenv(args.env) if args.env else load_dotenv()
    except ImportError:
        pass
    settings = Settings.from_env(generator="collage")
    from egig.llm import GeminiLLM
    from egig.orchestrator import make_vision

    if settings.gemini_api_key:
        llm = GeminiLLM(settings.gemini_api_key, settings.llm_model, settings.vlm_model, settings.llm_min_interval_s,
                        settings.llm_max_retries, settings.llm_temperature,
                        fallback_models=settings.llm_fallback_models)
    else:
        llm = NoLLM()
        if not args.reuse_vlm:
            print("[reverify] no GEMINI_API_KEY found: Gemini checks will be NOT VERIFIED (try --reuse-vlm or --env)")
    vision = make_vision(settings)
    if vision is None:
        print("[reverify] Grounding DINO is not available (pip install -r requirements-vision.txt)")
        return 2
    verifier = ReplayVerifier(llm, vision, settings, reuse_vlm=args.reuse_vlm)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    run_dirs = [d for d in find_runs(args.runs) if not only or d.name in only]
    if not run_dirs:
        print(f"[reverify] no runs under {args.runs}")
        return 1

    rows = []
    for d in run_dirs:
        old = load_run(d)
        print(f"[reverify] {d.name}: {len(old.reports)} attempt(s)")
        new = reverify_run(d, args.out / d.name, verifier, saved_checks=args.saved_checks)
        o = replay_loop(old.reports, settings.max_attempts) if old.reports else {}
        n = replay_loop(new.reports, settings.max_attempts)
        per_attempt = " ".join(
            f"{i}:{'P' if a.verdict.passed else 'F'}>{'P' if b.verdict.passed else 'F'}"
            for i, (a, b) in enumerate(zip(old.reports, new.reports), 1))
        rows.append({"id": d.name, "old_status": o.get("status", ""), "old_final": o.get("best_index", -1) + 1,
                     "new_status": n["status"], "new_final": n["best_index"] + 1,
                     "changed": "YES" if (o.get("status"), o.get("best_index")) != (n["status"], n["best_index"]) else "",
                     "needs_gpu_rerun": "YES" if n["incomplete"] else "", "attempts old>new": per_attempt})

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "reverify_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print()
    for r in rows:
        print(f"{r['id']:<24} {r['old_status']:>12} -> {r['new_status']:<12} {r['changed'] and 'CHANGED':<8} "
              f"{r['needs_gpu_rerun'] and 'needs GPU rerun':<16} {r['attempts old>new']}")
    print(f"\nGemini calls: {llm.calls} (reused saved answers for {verifier.reused} image(s))")
    print(f"wrote {args.out / 'reverify_summary.csv'}")
    print("needs GPU rerun = with the new verdicts the loop would have made another attempt that was never generated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
