"""Helper with automatic retry and dynamic model fallback for Gemini 503/429/404 errors."""

from __future__ import annotations

import time
from typing import Any, List, Optional
from google.genai.errors import APIError

_EXCLUDED_KEYWORDS = (
    "image",
    "tts",
    "embedding",
    "aqa",
    "imagen",
    "veo",
    "lyria",
    "robotics",
    "computer-use",
)


def _discover_gemini_models(client: Any) -> List[str]:
    discovered: List[str] = []
    try:
        for m in client.models.list():
            name = getattr(m, "name", "")
            if name.startswith("models/"):
                name = name[len("models/"):]
            lower = name.lower()
            if "gemini" in lower and not any(k in lower for k in _EXCLUDED_KEYWORDS):
                discovered.append(name)
    except Exception:
        pass
    return discovered


def generate_with_fallback(
    client: Any,
    primary_model: str,
    contents: Any,
    config: Any,
    max_attempts_per_model: int = 3,
) -> Any:
    candidates: List[str] = [primary_model]
    discovered_added = False
    idx = 0
    last_error: Optional[Exception] = None

    while idx < len(candidates):
        model_name = candidates[idx]
        idx += 1

        for attempt in range(1, max_attempts_per_model + 1):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
            except APIError as err:
                last_error = err
                code = getattr(err, "code", None)
                if code in (429, 500, 502, 503, 504):
                    wait_sec = 2 * attempt
                    print(
                        f"[Gemini Retry] {model_name} returned {code} "
                        f"(attempt {attempt}/{max_attempts_per_model}), waiting {wait_sec}s..."
                    )
                    time.sleep(wait_sec)
                    continue
                if code == 404:
                    print(f"[Gemini Fallback] Model '{model_name}' not found (404), trying next...")
                    break
                raise

        if not discovered_added:
            discovered_added = True
            for m in _discover_gemini_models(client):
                if m not in candidates:
                    candidates.append(m)

    if last_error is not None:
        raise last_error
    raise RuntimeError("All Gemini models failed to respond.")
