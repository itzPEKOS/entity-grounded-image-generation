"""Command line:

    python -m egig run --prompt "A Shiba Inu sitting in front of Wat Arun at sunset"
    python -m egig doctor          # check API key, models, GPU, internet
    python -m egig smoke           # load the GPU models once and make one test image (Kaggle)
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from .config import Settings


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def cmd_run(args: argparse.Namespace) -> int:
    from .llm import LLMError
    from .orchestrator import build_orchestrator

    settings = Settings.from_env(
        max_attempts=args.max_attempts, generator=args.generator, detector=args.detector,
        image_size=args.size, steps=args.steps, output_dir=args.output_dir,
    )
    try:
        result = build_orchestrator(settings).run(args.prompt)
    except LLMError as err:
        print(f"\n[error] {err}")
        return 2
    print(f"\nFinal image : {result.final_image}")
    print(f"Report      : {result.run_dir / 'final_report.md'}")
    return 0 if result.passed else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    s = Settings.from_env()
    ok = True
    print(f"Python {platform.python_version()} on {platform.system()}")

    print("\n[1] Gemini")
    if not s.gemini_api_key:
        print("  GEMINI_API_KEY is missing -> get one at https://aistudio.google.com/apikey, put it in .env")
        ok = False
    else:
        try:
            from pydantic import BaseModel

            from .llm import GeminiLLM

            llm = GeminiLLM(s.gemini_api_key, s.llm_model, s.vlm_model, min_interval_s=0,
                            fallback_models=s.llm_fallback_models)
            names = [n for n in llm.list_models() if "gemini" in n and ("flash" in n or "lite" in n)]
            print("  models your key can use (flash/lite):", ", ".join(names) or "(none listed)")
            print("  order tried when a model is busy:", " -> ".join(llm.fallback_chain(s.llm_model)))

            class Ping(BaseModel):
                ok: bool

            def answered(main: str) -> str:
                return "OK" if llm.last_model == main else f"busy, answered by fallback {llm.last_model}"

            llm.generate_json(Ping, ["Reply with ok=true."], system="Return JSON.")
            print(f"  LLM_MODEL={s.llm_model}: {answered(s.llm_model)}")
            if s.vlm_model != s.llm_model:
                llm.generate_json(Ping, ["Reply with ok=true."], system="Return JSON.", vision=True)
                print(f"  VLM_MODEL={s.vlm_model}: {answered(s.vlm_model)}")
        except Exception as err:
            print(f"  FAILED: {err}")
            ok = False

    print("\n[2] Wikipedia / Commons")
    if not s.wiki_contact:
        print("  WIKI_CONTACT is missing -> Wikimedia answers 403 without contact info. "
              "Put your email or a URL in .env")
        ok = False
    try:
        from .wiki import WikiClient

        wiki = WikiClient(s.wiki_contact, s.wiki_lang)
        page = wiki.find_page("Eiffel Tower")
        imgs = wiki.commons_images("Eiffel Tower", limit=3)
        print(f"  page: {page.title if page else None}, lead image: {bool(page and page.image_url)}, commons: {len(imgs)}")
        ok = ok and page is not None
    except Exception as err:
        print(f"  FAILED: {err}")
        ok = False

    print("\n[3] Vision / GPU")
    try:
        import torch

        print(f"  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"   - cuda:{i} {p.name} {p.total_memory / 2**30:.1f} GB")
    except ImportError:
        print("  torch not installed (OK on a laptop: detector checks are skipped, generator = collage)")
    for mod in ("transformers", "diffusers", "accelerate", "cv2"):
        try:
            m = __import__(mod)
            print(f"  {mod} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"  {mod}: not installed")
    print("\nAll good." if ok else "\nSome checks failed (see above).")
    return 0 if ok else 1


def cmd_smoke(args: argparse.Namespace) -> int:
    """Load SDXL + IP-Adapter + ControlNet and Grounding DINO, generate one image, detect in it."""
    from PIL import Image

    from .backends import GenerationJob
    from .backends.sdxl import SDXLBackend
    from .vision import VisionTools, default_vision_device

    s = Settings.from_env()
    out = Path(s.output_dir) / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    backend = SDXLBackend(s, device="cuda:0")
    size = args.size
    job = GenerationJob(prompt="a photo of a dog sitting on a beach, sharp focus", negative_prompt="blurry",
                        width=size, height=size, steps=args.steps, guidance=6.0, seed=1)
    img = backend.generate(job)
    img.save(out / "smoke_sdxl.png")
    print(f"[smoke] SDXL OK -> {out / 'smoke_sdxl.png'}")

    ref = img.crop((size // 4, size // 4, 3 * size // 4, 3 * size // 4))
    mask = Image.new("L", (size, size), 0)
    mask.paste(255, (size // 4, size // 4, 3 * size // 4, 3 * size // 4))
    job2 = GenerationJob(prompt="a photo of a dog in a park", negative_prompt="blurry", width=size, height=size,
                         steps=args.steps, guidance=6.0, seed=2, ip_images=[ref], ip_masks=[mask], ip_scales=[0.6])
    backend.generate(job2).save(out / "smoke_ip_adapter.png")
    print(f"[smoke] IP-Adapter with mask OK -> {out / 'smoke_ip_adapter.png'}")

    vision = VisionTools(s.gdino_model, s.clip_model, device=default_vision_device())
    dets = vision.detect(img, "dog")
    print(f"[smoke] Grounding DINO on {vision.device}: {[(round(d.score, 2), d.box) for d in dets]}")
    return 0


def main(argv: list | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Thai text on Windows consoles
    _load_env()
    parser = argparse.ArgumentParser(prog="python -m egig", description="Entity-grounded image generation")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full multi-agent pipeline")
    run.add_argument("--prompt", "-p", required=True)
    run.add_argument("--max-attempts", type=int)
    run.add_argument("--generator", choices=["auto", "sdxl", "collage"])
    run.add_argument("--detector", choices=["auto", "gdino", "none"])
    run.add_argument("--size", type=int, help="image width/height in pixels (default 1024)")
    run.add_argument("--steps", type=int, help="denoising steps (default 25)")
    run.add_argument("--output-dir")
    run.set_defaults(func=cmd_run)

    doctor = sub.add_parser("doctor", help="check API key, models, internet and GPU")
    doctor.set_defaults(func=cmd_doctor)

    smoke = sub.add_parser("smoke", help="GPU smoke test: SDXL, IP-Adapter mask, Grounding DINO")
    smoke.add_argument("--size", type=int, default=768)
    smoke.add_argument("--steps", type=int, default=12)
    smoke.set_defaults(func=cmd_smoke)

    args = parser.parse_args(argv)
    return args.func(args)
