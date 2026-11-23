"""LLM/VLM access. Agents only depend on the small `LLM` protocol, so tests can plug in a
fake and the real implementation (Gemini) can be swapped later."""

from __future__ import annotations

import io
import json
import re
import time
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, Type, TypeVar, Union

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


def _short(err: Exception, limit: int = 80) -> str:
    """The server's message, e.g. 'The model is overloaded.', so the log shows why."""
    msg = getattr(err, "message", None) or str(err)
    msg = " ".join(str(msg).split())
    return f"({msg[:limit]}{'...' if len(msg) > limit else ''})"


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.S)
    return m.group(1) if m else text


class _ModelUnavailable(Exception):
    """One model stayed busy (5xx) or out of quota (429) after its retries: try the next one."""

    def __init__(self, model: str, code: Optional[int], err: Exception):
        super().__init__(f"{model}: HTTP {code}")
        self.model, self.code, self.err = model, code, err


_AUTO_FALLBACK = re.compile(r"^gemini-(\d+(?:\.\d+)?)-flash(-lite)?$")


class GeminiLLM:
    """Google Gemini via the `google-genai` SDK, with JSON-schema output, a simple rate
    limiter (free tier is ~15 requests/minute) and retries on 429/5xx.

    Fallback: when a model stays overloaded (HTTP 503) or out of free quota (429), the call moves
    on to the next model in `fallback_models` ("auto" = the stable gemini-X-flash / -flash-lite
    models your key can list). Free-tier quotas are per model, so this also helps with 429."""

    RETRY_CODES = (429, 500, 502, 503, 504)
    STICKY_S = 600.0  # after a fallback worked, keep using it for 10 minutes before trying the main model again

    def __init__(self, api_key: str, model: str, vlm_model: Optional[str] = None,
                 min_interval_s: float = 4.5, max_retries: int = 5, temperature: float = 0.2,
                 fallback_models: Union[str, Sequence[str]] = "auto"):
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
        self.fallback_models = fallback_models
        self.last_model = ""  # the model that answered the last call
        self._fallbacks: Optional[List[str]] = None  # resolved on first need
        self._sticky: Dict[str, Tuple[str, float]] = {}  # main model -> (fallback that worked, until)

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
        main = self.vlm_model if vision else self.model
        response = self._call_with_fallback(main, contents, config)
        return self._parse(schema, response)

    def list_models(self) -> List[str]:
        names = []
        for m in self.client.models.list():
            name = getattr(m, "name", "") or ""
            names.append(name.removeprefix("models/"))
        return sorted(names)

    def fallback_chain(self, main: str) -> List[str]:
        """Models tried in order for a call whose configured model is `main`."""
        chain = [main] + [m for m in self._resolve_fallbacks() if m != main]
        sticky = self._sticky.get(main)
        if sticky and time.monotonic() < sticky[1] and sticky[0] in chain:
            chain.remove(sticky[0])
            chain.insert(0, sticky[0])
        return chain

    # ----------------------------------------------------------------- private
    def _resolve_fallbacks(self) -> List[str]:
        if self._fallbacks is not None:
            return self._fallbacks
        spec = self.fallback_models
        if not isinstance(spec, str):
            self._fallbacks = [m.strip() for m in spec if m.strip()]
        elif spec.strip().lower() in ("", "none", "off"):
            self._fallbacks = []
        elif spec.strip().lower() != "auto":
            self._fallbacks = [m.strip() for m in spec.split(",") if m.strip()]
        else:
            try:
                names = self.list_models()
            except Exception as err:  # e.g. the API is down too: try again on the next failure
                print(f"[gemini] could not list models for fallback: {str(err)[:120]}")
                return []
            found = []
            for n in names:
                m = _AUTO_FALLBACK.match(n)
                if m:
                    found.append((-float(m.group(1)), bool(m.group(2)), n))  # newest first, flash before lite
            self._fallbacks = [n for *_, n in sorted(found)][:4]
            print(f"[gemini] fallback models: {', '.join(self._fallbacks) or '(none found)'}")
        return self._fallbacks

    def _call_with_fallback(self, main: str, contents: List[object], config: object) -> object:
        chain = self.fallback_chain(main)
        failures: List[str] = []
        i = 0
        while i < len(chain):
            model = chain[i]
            # with somewhere to fall back to, don't wait long on one model
            tries = self.max_retries + 1 if len(chain) == 1 else min(self.max_retries, 2) + 1
            try:
                response = self._call(model, contents, config, tries, is_main=(model == main),
                                      can_fall_back=i + 1 < len(chain))
            except _ModelUnavailable as busy:
                failures.append(f"{busy.model} (HTTP {busy.code})")
                if len(chain) == 1 and self._fallbacks is None:  # listing models failed earlier: try once more
                    chain += [m for m in self._resolve_fallbacks() if m not in chain]
                if i + 1 < len(chain):
                    print(f"[gemini] {busy.model} unavailable (HTTP {busy.code}), switching to {chain[i + 1]}")
                i += 1
                continue
            self.last_model = model
            if model != main:
                self._sticky[main] = (model, time.monotonic() + self.STICKY_S)
            elif main in self._sticky:
                del self._sticky[main]
            return response
        if failures and all("429" in f for f in failures):
            raise LLMError(
                "Gemini quota exhausted (HTTP 429) on every model tried: " + ", ".join(failures) +
                ". The free tier has per-minute and per-day limits per model; wait and run again.")
        raise LLMError(
            "Gemini is unavailable right now (overloaded or out of quota) on every model tried: " +
            ", ".join(failures) + ". This is on Google's side; wait a few minutes and run again, or set "
            "LLM_FALLBACK_MODELS to other models from `python -m egig doctor`.")

    def _call(self, model: str, contents: List[object], config: object, tries: int, is_main: bool,
              can_fall_back: bool = False) -> object:
        from google.genai import errors

        for attempt in range(tries):
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
                    if not is_main:  # a fallback that this key cannot use: just skip it
                        raise _ModelUnavailable(model, code, err) from err
                    raise LLMError(
                        f"Gemini model '{model}' was not found. Run `python -m egig doctor` to list the "
                        "models your key can use, then set LLM_MODEL / VLM_MODEL in .env."
                    ) from err
                if code in (400, 401, 403) and "key" in str(err).lower():
                    raise LLMError(f"Gemini rejected the API key: {err}") from err
                if code in self.RETRY_CODES:
                    # quotas are per model, so on 429 another model is faster than waiting 20 s+
                    if attempt + 1 < tries and not (code == 429 and can_fall_back):
                        delay = self._retry_delay(err, code, attempt)
                        print(f"[gemini] {model}: HTTP {code} {_short(err)}, retrying in {delay:.0f}s "
                              f"({attempt + 1}/{tries - 1})")
                        time.sleep(delay)
                        continue
                    raise _ModelUnavailable(model, code, err) from err
                raise LLMError(f"Gemini error (HTTP {code}) from {model}: {err}") from err
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
