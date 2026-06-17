"""Tests for the BytePlus multimodal embedding adapter.

BytePlus's only available embedding models (``skylark-embedding-vision-*``) do
NOT work on the OpenAI-compatible ``/embeddings`` batch endpoint. They require
``/embeddings/multimodal`` with a *typed* ``input`` array, and each request
fuses its ``input`` parts into a SINGLE vector (verified against the live API).

So embedding N text chunks => N separate POSTs, one vector each, returned in the
original order.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from deeptutor.services.embedding.adapters.base import (
    EmbeddingProviderError,
    EmbeddingRequest,
)
from deeptutor.services.embedding.adapters.byteplus_multimodal import (
    BytePlusMultiModalEmbeddingAdapter,
)

BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/embeddings/multimodal"
MODEL = "skylark-embedding-vision-251215"


def _req(texts: list[str]) -> EmbeddingRequest:
    return EmbeddingRequest(texts=texts, model=MODEL)


def _make_adapter(**overrides: Any) -> BytePlusMultiModalEmbeddingAdapter:
    config = {
        "api_key": "ark-test-key",
        "base_url": BASE_URL,
        "model": MODEL,
        "dimensions": 2048,
        "request_timeout": 5,
    }
    config.update(overrides)
    return BytePlusMultiModalEmbeddingAdapter(config)


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Patch ``httpx.AsyncClient.post`` to record calls and echo a vector whose
    first component encodes the input text length (so result ORDER is testable)."""
    calls: list[dict[str, Any]] = []

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        body = kwargs.get("json")
        calls.append({"url": url, "json": body, "headers": kwargs.get("headers")})
        text = body["input"][0]["text"]
        vec = [float(len(text)), 0.0, 0.0]
        request = httpx.Request("POST", url)
        return httpx.Response(
            status_code=200,
            json={
                "object": "list",
                "model": "skylark-embedding-vision-251215",
                "data": {"embedding": vec},
            },
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return calls


@pytest.mark.asyncio
async def test_one_request_per_text_preserving_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture(monkeypatch)
    adapter = _make_adapter()

    resp = await adapter.embed(_req(["aa", "bbbb", "c"]))

    assert len(calls) == 3
    assert resp.embeddings == [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


@pytest.mark.asyncio
async def test_posts_typed_text_input_to_verbatim_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture(monkeypatch)
    adapter = _make_adapter()

    await adapter.embed(_req(["hello world"]))

    assert len(calls) == 1
    assert calls[0]["url"] == BASE_URL
    assert calls[0]["json"]["model"] == "skylark-embedding-vision-251215"
    assert calls[0]["json"]["input"] == [{"type": "text", "text": "hello world"}]


@pytest.mark.asyncio
async def test_sends_bearer_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture(monkeypatch)
    adapter = _make_adapter(api_key="ark-secret")

    await adapter.embed(_req(["x"]))

    assert calls[0]["headers"]["Authorization"] == "Bearer ark-secret"


@pytest.mark.asyncio
async def test_parses_data_as_list_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Be defensive: if the API ever returns ``data`` as a list, take the first."""

    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(
            status_code=200,
            json={"data": [{"embedding": [9.0, 8.0, 7.0]}]},
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    adapter = _make_adapter()

    resp = await adapter.embed(_req(["only"]))

    assert resp.embeddings == [[9.0, 8.0, 7.0]]


@pytest.mark.asyncio
async def test_http_error_raises_embedding_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(
            status_code=404,
            json={"error": {"code": "InvalidEndpointOrModel.NotFound", "message": "nope"}},
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    adapter = _make_adapter()

    with pytest.raises(EmbeddingProviderError):
        await adapter.embed(_req(["x"]))


@pytest.mark.asyncio
async def test_empty_texts_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture(monkeypatch)
    adapter = _make_adapter()

    resp = await adapter.embed(_req([]))

    assert resp.embeddings == []
    assert calls == []


def test_model_info_reports_dimension_and_provider() -> None:
    adapter = _make_adapter()
    info = adapter.get_model_info()
    assert info["model"] == "skylark-embedding-vision-251215"
    assert info["dimensions"] == 2048
    assert info["provider"] == "byteplus"


# --- Wiring: provider spec registered + client resolves the adapter ----------


def test_byteplus_embedding_spec_registered() -> None:
    from deeptutor.services.config.provider_runtime import EMBEDDING_PROVIDERS

    spec = EMBEDDING_PROVIDERS["byteplus"]
    assert spec.adapter == "byteplus_multimodal"
    assert spec.default_model == "skylark-embedding-vision-251215"
    assert spec.default_dim == 2048
    assert "embeddings/multimodal" in spec.default_api_base


def test_client_resolves_byteplus_binding_to_adapter() -> None:
    from deeptutor.services.embedding.client import _resolve_adapter_class

    assert _resolve_adapter_class("byteplus") is BytePlusMultiModalEmbeddingAdapter


def test_byteplus_multimodal_url_passes_endpoint_validation() -> None:
    """The ``/embeddings/multimodal`` URL must NOT be rejected for ending in
    ``/multimodal`` instead of ``/embeddings``."""
    from deeptutor.services.config.embedding_endpoint import (
        embedding_endpoint_validation_error,
    )

    err = embedding_endpoint_validation_error("byteplus", BASE_URL)
    assert err is None
