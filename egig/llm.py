"""LLM/VLM access. Agents only depend on the small `LLM` protocol, so tests can plug in a
fake and the real implementation (Gemini) can be swapped later."""

from __future__ import annotations

import io
import json
import re
import time
from typing import List, Optional, Protocol, Sequence, Type, TypeVar, Union

from PIL import Image
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
Part = Union[str, Image.Image]


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    calls: int

    def generate_json(self, schema: Type[T], parts: Sequence[Part], *, system: str, vision: bool = False) -> T:
        """Send text and images, get back an instance of `schema`."""
        ...


def _image_to_jpeg(img: Image.Image, max_side: int = 768) -> bytes:
    img = img.convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.S)
    return m.group(1) if m else text


class GeminiLLM:
    """Google Gemini via the `google-genai` SDK, with JSON-schema output, a simple rate
    limiter (free tier is ~15 requests/minute) and retries on 429/5xx."""

    RETRY_CODES = (429, 500, 502, 503, 504)

    def __init__(self, api_key: str, model: str, vlm_model: Optional[str] = None,
                 min_interval_s: float = 4.5, max_retries: int = 5, temperature: float = 0.2):
        if not api_key:
            raise LLMError(
                "GEMINI_API_KEY is empty. Get a free key at https://aistudio.google.com/apikey "
                "and put it in .env (laptop) or Kaggle Secrets (Kaggle)."
            )
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.vlm_model = vlm_model or model
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.temperature = temperature
        self.calls = 0
        self._last_call = 0.0

    # ------------------------------------------------------------------ public
    def generate_json(self, schema: Type[T], parts: Sequence[Part], *, system: str, vision: bool = False) -> T:
        from google.genai import types

        contents: List[object] = []
        for p in parts:
            if isinstance(p, Image.Image):
                contents.append(types.Part.from_bytes(data=_image_to_jpeg(p), mime_type="image/jpeg"))
            else:
                contents.append(str(p))
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=self.temperature,
        )
        model = self.vlm_model if vision else self.model
        response = self._call(model, contents, config)
        return self._parse(schema, response)

    def list_models(self) -> List[str]:
        names = []
        for m in self.client.models.list():
            name = getattr(m, "name", "") or ""
            names.append(name.removeprefix("models/"))
        return sorted(names)

    # ----------------------------------------------------------------- private
    def _call(self, model: str, contents: List[object], config: object) -> object:
        from google.genai import errors

        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                self.calls += 1
                return self.client.models.generate_content(model=model, contents=contents, config=config)
            except errors.APIError as err:
                code = getattr(err, "code", None)
                if code == 404:
                    raise LLMError(
                        f"Gemini model '{model}' was not found. Run `python -m egig doctor` to list the "
                        "models your key can use, then set LLM_MODEL / VLM_MODEL in .env."
                    ) from err
                if code in (400, 401, 403) and "key" in str(err).lower():
                    raise LLMError(f"Gemini rejected the API key: {err}") from err
                if code in self.RETRY_CODES and attempt < self.max_retries:
                    delay = self._retry_delay(err, code, attempt)
                    print(f"[gemini] HTTP {code}, retrying in {delay:.0f}s ({attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue
                if code == 429:
                    raise LLMError(
                        "Gemini quota exhausted (HTTP 429). The free tier has per-minute and per-day "
                        f"limits; wait, or switch LLM_MODEL/VLM_MODEL to another model. Details: {err}"
                    ) from err
                raise
        raise LLMError("unreachable")

    @staticmethod
    def _retry_delay(err: Exception, code: Optional[int], attempt: int) -> float:
        m = re.search(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?(\d+(?:\.\d+)?)s", str(err), flags=re.I)
        if m:
            return min(float(m.group(1)) + 1, 90.0)
        return min((20.0 if code == 429 else 4.0) * (attempt + 1), 90.0)

    @staticmethod
    def _parse(schema: Type[T], response: object) -> T:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        text = getattr(response, "text", None) or ""
        try:
            return schema.model_validate_json(_strip_fences(text))
        except ValidationError as err:
            try:  # sometimes the model wraps the object in a list
                data = json.loads(_strip_fences(text))
                if isinstance(data, list) and data:
                    return schema.model_validate(data[0])
            except (json.JSONDecodeError, ValidationError):
                pass
            raise LLMError(f"Gemini returned JSON that does not match {schema.__name__}: {text[:500]}") from err
