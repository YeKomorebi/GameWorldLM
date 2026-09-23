import json

import httpx
import pytest
from openai import OpenAI

from llm.providers import LLMError, OpenAICompatibleBackend, ProviderConfig


def _client(handler):
    return OpenAI(
        api_key="test-key-not-real",
        base_url="https://unit-test.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _completion(content="{}", reason="stop", refusal=None):
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": reason,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
    }


@pytest.mark.parametrize(
    "provider,mode,budget_key",
    [
        ("openai", "json_schema", "max_completion_tokens"),
        ("qwen", "json_object", "max_tokens"),
    ],
)
def test_real_sdk_serializes_provider_contract(provider, mode, budget_key, forest):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json=_completion(forest.model_dump_json()))

    backend = OpenAICompatibleBackend(
        ProviderConfig(provider, "test-model", "hidden", "unused"), _client(handler)
    )
    try:
        assert (
            backend.complete([{"role": "user", "content": "Return JSON"}])
            == forest.model_dump_json()
        )
        assert requests[0]["response_format"]["type"] == mode
        assert requests[0][budget_key] == 8192
        assert requests[0]["model"] == "test-model"
        if provider == "qwen":
            assert requests[0]["enable_thinking"] is False
        else:
            assert "enable_thinking" not in requests[0]
    finally:
        backend.close()


@pytest.mark.parametrize(
    "response,match",
    [
        (_completion(reason="length"), "Incomplete"),
        (_completion(refusal="No"), "refused"),
        (_completion(content=""), "empty"),
        (
            {"id": "x", "choices": [], "object": "chat.completion", "created": 0, "model": "x"},
            "no completion",
        ),
    ],
)
def test_incomplete_or_refused_output_is_rejected(response, match):
    backend = OpenAICompatibleBackend(
        ProviderConfig("openai", "model", "hidden", "unused"),
        _client(lambda request: httpx.Response(200, json=response)),
    )
    try:
        with pytest.raises(LLMError, match=match):
            backend.complete([])
    finally:
        backend.close()


def test_auth_failure_does_not_expose_provider_body():
    backend = OpenAICompatibleBackend(
        ProviderConfig("qwen", "model", "hidden", "unused"),
        _client(lambda request: httpx.Response(401, json={"error": {"message": "secret-value"}})),
    )
    try:
        with pytest.raises(LLMError) as exc:
            backend.complete([])
        assert "401" in str(exc.value)
        assert "secret-value" not in str(exc.value)
        assert "hidden" not in repr(backend.config)
    finally:
        backend.close()


def test_missing_key_has_actionable_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing OPENAI_API_KEY"):
        ProviderConfig.from_env("openai")


def test_response_metadata_is_captured_and_reset_after_error():
    responses = iter(
        [
            httpx.Response(
                200,
                headers={"x-request-id": "trace-1"},
                json=_completion()
                | {
                    "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
                },
            ),
            httpx.Response(429, json={"error": {"message": "quota"}}),
        ]
    )
    backend = OpenAICompatibleBackend(
        ProviderConfig("qwen", "requested", "hidden", "unused"),
        _client(lambda request: next(responses)),
    )
    try:
        backend.complete([])
        assert backend.last_metadata["actual_model"] == "test-model"
        assert backend.last_metadata["usage"]["total_tokens"] == 11
        assert backend.last_metadata["request_id"] == "trace-1"
        with pytest.raises(LLMError):
            backend.complete([])
        assert backend.last_metadata["http_status"] == 429
        assert "usage" not in backend.last_metadata
    finally:
        backend.close()
