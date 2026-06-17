"""BytePlus (ModelArk) multimodal embedding adapter.

BytePlus's available embedding models (``skylark-embedding-vision-*``) only work
through the native ``/embeddings/multimodal`` endpoint with a *typed* ``input``
array (``[{"type": "text", "text": ...}, {"type": "image_url", ...}]``), NOT the
OpenAI-compatible ``/embeddings`` batch endpoint.

Verified against the live API: one request fuses its ``input`` parts into a
SINGLE vector. There is no native batching — embedding N text chunks means N
separate POSTs, one vector each. We fan those out concurrently (bounded) and
return the vectors in the original order.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

import httpx

from deeptutor.services.llm.openai_http_client import disable_ssl_verify_enabled

from .base import (
    BaseEmbeddingAdapter,
    EmbeddingProviderError,
    EmbeddingRequest,
    EmbeddingResponse,
)

logger = logging.getLogger(__name__)


class BytePlusMultiModalEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for BytePlus ModelArk multimodal (vision) embedding models."""

    PROVIDER = "byteplus"
    DEFAULT_DIM = 2048
    # Max concurrent in-flight requests, since each text needs its own POST.
    MAX_CONCURRENCY = 8

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = str(self.api_key or "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        headers.update({str(k): str(v) for k, v in self.extra_headers.items()})
        return headers

    @staticmethod
    def _extract_vector(payload: Any) -> List[float]:
        """Pull the single embedding vector out of a multimodal response.

        Live shape: ``{"data": {"embedding": [...]}, ...}`` (``data`` is an
        object, not a list). We also accept a list-shaped ``data`` defensively.
        """
        if not isinstance(payload, dict):
            raise ValueError(f"Embedding response is not a JSON object: {type(payload).__name__}")
        if "error" in payload:
            raise ValueError(f"Embedding provider returned error payload: {payload['error']}")

        data = payload.get("data")
        if isinstance(data, dict):
            vec = data.get("embedding")
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            vec = data[0].get("embedding")
        else:
            vec = payload.get("embedding")

        if not isinstance(vec, list):
            keys = sorted(payload.keys())
            raise ValueError(f"Cannot find embedding vector in response. Top-level keys={keys}.")
        return [float(x) for x in vec]

    async def _embed_one(
        self,
        client: httpx.AsyncClient,
        model: str,
        text: str,
        headers: Dict[str, str],
    ) -> List[float]:
        payload = {"model": model, "input": [{"type": "text", "text": text}]}
        response = await client.post(self.base_url, json=payload, headers=headers)
        if response.status_code >= 400:
            body_text = response.text
            logger.error(f"HTTP {response.status_code} from {self.base_url}: {body_text[:2000]}")
            raise EmbeddingProviderError(
                f"Embedding provider returned HTTP {response.status_code}",
                status=response.status_code,
                body=body_text,
                model=model,
                url=self.base_url,
                provider=self.PROVIDER,
            )
        return self._extract_vector(response.json())

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        texts = list(request.texts or [])
        model = request.model or self.model
        if not texts:
            return EmbeddingResponse(embeddings=[], model=model, dimensions=0, usage={})

        headers = self._headers()
        timeout = httpx.Timeout(
            connect=10.0,
            read=max(self.request_timeout, 60),
            write=10.0,
            pool=10.0,
        )
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENCY)

        async with httpx.AsyncClient(
            timeout=timeout, verify=not disable_ssl_verify_enabled()
        ) as client:

            async def run(text: str) -> List[float]:
                async with semaphore:
                    return await self._embed_one(client, model, text, headers)

            embeddings = await asyncio.gather(*(run(t) for t in texts))

        actual_dims = len(embeddings[0]) if embeddings else 0
        logger.info(
            f"Successfully generated {len(embeddings)} BytePlus embeddings "
            f"(model: {model}, dimensions: {actual_dims})"
        )
        return EmbeddingResponse(
            embeddings=list(embeddings),
            model=model,
            dimensions=actual_dims,
            usage={},
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "dimensions": self.dimensions or self.DEFAULT_DIM,
            "supports_variable_dimensions": False,
            "multimodal": False,
            "provider": self.PROVIDER,
        }
