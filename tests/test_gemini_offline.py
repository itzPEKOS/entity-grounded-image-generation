"""Run the real google-genai SDK with its HTTP call replaced, so we test request building
(schema conversion, images) and response parsing without an API key or network."""

import json

import pytest
from PIL import Image

genai = pytest.importorskip("google.genai")
from google.genai import errors, types  # noqa: E402

from egig.llm import GeminiLLM, LLMError  # noqa: E402
from egig.schemas import ParsedQuery, VisualProfile, VlmVerdicts  # noqa: E402


def wire(monkeypatch, replies, fallback_models="none"):
    """`replies`: list of dicts (JSON answers) or exceptions, consumed in order."""
    llm = GeminiLLM(api_key="test-key", model="gemini-test", vlm_model="gemini-vlm-test", min_interval_s=0,
                    fallback_models=fallback_models)
    sent = []

    def fake_request(http_method, path, request_dict, http_options=None):
        sent.append((path, request_dict))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        body = {"candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(reply)}]},
                                "finishReason": "STOP"}]}
        return types.HttpResponse(headers={}, body=json.dumps(body))

    monkeypatch.setattr(llm.client._api_client, "request", fake_request)
    monkeypatch.setattr("egig.llm.time.sleep", lambda s: None)
    return llm, sent


def test_parsed_query_schema_round_trip(monkeypatch, parsed_query):
    llm, sent = wire(monkeypatch, [json.loads(parsed_query.model_dump_json())])
    out = llm.generate_json(ParsedQuery, ["User request: A Shiba Inu"], system="sys")
    assert isinstance(out, ParsedQuery) and out.entities[1].attributes == ["sitting"]
    path, body = sent[0]
    assert "gemini-test" in path
    gen_cfg = body.get("generationConfig") or body.get("generation_config")
    assert gen_cfg["responseMimeType"] == "application/json"
    schema_text = json.dumps(gen_cfg)
    assert "detection_label" in schema_text and "landmark" in schema_text  # fields + enum values made it


def test_images_are_sent_and_vlm_model_is_used(monkeypatch):
    reply = {"verdicts": [{"check_id": "a", "passed": True, "confidence": 0.9, "evidence": "ok", "fix_hint": ""}]}
    llm, sent = wire(monkeypatch, [reply])
    out = llm.generate_json(VlmVerdicts, ["IMAGE A:", Image.new("RGB", (1000, 500)), "Checks: a"],
                            system="sys", vision=True)
    assert out.verdicts[0].check_id == "a"
    path, body = sent[0]
    assert "gemini-vlm-test" in path
    parts = body["contents"][0]["parts"]  # all parts grouped into ONE user turn
    assert len(parts) == 3 and "inlineData" in parts[1]
    inline = parts[1]["inlineData"]
    assert (inline.get("mimeType") or inline.get("mime_type")) == "image/jpeg"


def test_retry_on_429_then_success(monkeypatch):
    profile = {"summary": "s", "key_features": ["f"], "negative_traits": [], "confusables": [], "prompt_keywords": "k"}
    busy = errors.ClientError(429, {"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}})
    llm, sent = wire(monkeypatch, [busy, profile])
    out = llm.generate_json(VisualProfile, ["x"], system="sys")
    assert out.key_features == ["f"] and len(sent) == 2 and llm.calls == 2


def test_unknown_model_gives_a_helpful_error(monkeypatch):
    missing = errors.ClientError(404, {"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}})
    llm, _ = wire(monkeypatch, [missing])
    with pytest.raises(LLMError, match="doctor"):
        llm.generate_json(VisualProfile, ["x"], system="sys")


PROFILE = {"summary": "s", "key_features": ["f"], "negative_traits": [], "confusables": [], "prompt_keywords": "k"}


def overloaded():
    return errors.ServerError(503, {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}})


def models_called(sent):
    return [path.split("models/")[1].split(":")[0] for path, _ in sent]


def test_overloaded_model_falls_back_and_sticks(monkeypatch):
    llm, sent = wire(monkeypatch, [overloaded(), overloaded(), overloaded(), PROFILE, PROFILE],
                     fallback_models="gemini-backup")
    out = llm.generate_json(VisualProfile, ["x"], system="sys")
    assert out.key_features == ["f"]
    # 3 tries on the main model (1 + 2 retries), then the fallback answers
    assert models_called(sent) == ["gemini-test"] * 3 + ["gemini-backup"]
    assert llm.last_model == "gemini-backup"
    # next call goes straight to the fallback that worked (no more waiting on the busy model)
    llm.generate_json(VisualProfile, ["y"], system="sys")
    assert models_called(sent)[-1] == "gemini-backup" and len(sent) == 5


def test_every_model_down_raises_llm_error(monkeypatch):
    llm, sent = wire(monkeypatch, [overloaded() for _ in range(6)], fallback_models="gemini-backup")
    with pytest.raises(LLMError, match="unavailable"):
        llm.generate_json(VisualProfile, ["x"], system="sys")
    assert len(sent) == 6  # 3 per model, then give up instead of retrying forever


def test_missing_fallback_model_is_skipped(monkeypatch):
    gone = errors.ClientError(404, {"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}})
    llm, sent = wire(monkeypatch, [overloaded(), overloaded(), overloaded(), gone, PROFILE],
                     fallback_models="gemini-gone,gemini-backup")
    llm.generate_json(VisualProfile, ["x"], system="sys")
    assert models_called(sent)[-2:] == ["gemini-gone", "gemini-backup"]


def test_auto_fallback_picks_stable_flash_models(monkeypatch):
    llm, _ = wire(monkeypatch, [], fallback_models="auto")
    monkeypatch.setattr(llm, "list_models", lambda: [
        "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash", "gemini-3.5-flash-lite",
        "gemini-3.5-flash-preview-09", "gemini-3.5-pro", "gemini-2.5-flash-image", "text-embedding-004"])
    assert llm.fallback_chain("gemini-3.5-flash-lite") == [
        "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]


def test_quota_429_switches_model_without_waiting(monkeypatch):
    quota = errors.ClientError(429, {"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}})
    llm, sent = wire(monkeypatch, [quota, PROFILE], fallback_models="gemini-backup")
    llm.generate_json(VisualProfile, ["x"], system="sys")
    assert models_called(sent) == ["gemini-test", "gemini-backup"]  # no 20 s retries on the main model
