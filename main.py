"""CLI entrypoint for Entity-Grounded Image Generation."""

from __future__ import annotations

import argparse
import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> bool:
        return False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False

from src.agents import (
    EntityExtractorAgent,
    ImageGeneratorAgent,
    VerificationAgent,
    VisualLookupAgent,
)
from src.orchestrator import EntityGroundedOrchestrator


def build_orchestrator(demo: bool, max_retries: int, output_dir: str) -> EntityGroundedOrchestrator:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if demo or not api_key:
        if not demo and not api_key:
            print("No GEMINI_API_KEY detected — running in offline demo mode.")
        return EntityGroundedOrchestrator(
            max_retries=max_retries,
            output_root=output_dir,
        )

    from google import genai

    client = genai.Client(api_key=api_key)
    llm_model = os.getenv("LLM_MODEL", "gemini-3.8-flash")
    vlm_model = os.getenv("VLM_MODEL", "gemini-3.8-flash")
    image_model = os.getenv("IMAGE_MODEL", "flux")

    return EntityGroundedOrchestrator(
        extractor=EntityExtractorAgent(llm_client=client, model=llm_model),
        lookup=VisualLookupAgent(llm_client=client, model=llm_model),
        generator=ImageGeneratorAgent(image_client=client, model=image_model),
        verifier=VerificationAgent(vlm_client=client, model=vlm_model),
        max_retries=max_retries,
        output_root=output_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Entity-Grounded Image Generation Multi-Agent System"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Text-to-image query (e.g., 'A Shiba Inu sitting in front of Wat Arun at sunset')",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum self-correction retries if verification checks fail (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where images, constraint specs, and reports are saved",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in offline demo mode without making external API calls",
    )
    args = parser.parse_args()

    orchestrator = build_orchestrator(
        demo=args.demo,
        max_retries=args.max_retries,
        output_dir=args.output_dir,
    )
    result = orchestrator.run(args.prompt)

    if HAS_RICH:
        console = Console()
        console.print(
            Panel.fit(
                f"[bold cyan]Entity-Grounded Image Generation[/bold cyan]\nPrompt: [white]{args.prompt}[/white]",
                border_style="cyan",
            )
        )
        table = Table(title=f"Final Verification Report (Attempt {result.final_report.attempt})")
        table.add_column("Check ID", style="cyan")
        table.add_column("Entity", style="magenta")
        table.add_column("Criterion")
        table.add_column("Result", justify="center")
        table.add_column("Evidence")

        for v in result.final_report.verdicts:
            status = "[bold green]PASS[/bold green]" if v.passed else "[bold red]FAIL[/bold red]"
            table.add_row(v.check_id, v.entity_id or "global", v.criterion, status, v.observed_evidence)

        console.print(table)
    else:
        print(result.final_report.to_markdown())

    print(f"Artifacts saved to: {result.run_dir}")
    print(f"Final Image: {result.final_image_path}")
    print(f"Overall Status: {'PASS' if result.final_report.overall_passed else 'FAIL'}")

    return 0 if result.final_report.overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
