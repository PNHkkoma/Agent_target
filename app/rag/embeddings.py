from __future__ import annotations

import hashlib
import math
import re
import unicodedata
import asyncio
from abc import ABC, abstractmethod

import httpx


# Contract chung để đổi embedding backend mà ingestion/retrieval không phải sửa.
class EmbeddingProvider(ABC):
    provider_name: str
    model_name: str
    dimensions: int
    version: str

    # Nhận nhiều đoạn text; trả một vector cùng thứ tự cho mỗi đoạn.
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    # Không nhận đầu vào; đóng tài nguyên mạng nếu provider có sử dụng.
    async def close(self) -> None:
        return None


# Baseline embedding offline dựa trên token hash, dùng cho test chứ không thay semantic model.
class LocalHashEmbeddingProvider(EmbeddingProvider):
    provider_name = "test_local_hash"
    model_name = "local-hash-baseline-v1"
    version = "1"

    # Nhận số chiều; tạo provider deterministic không cần model hoặc API key.
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    # Nhận text; trả token tiếng Việt/Anh đã lower-case và bỏ dấu để hashing ổn định.
    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFD", text.casefold())
            if unicodedata.category(character) != "Mn"
        ).replace("đ", "d")
        return re.findall(r"[a-z0-9]+", normalized)

    # Nhận nhiều đoạn text; trả vector L2-normalized theo token và bigram đã hash.
    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            tokens = self._tokens(text)
            features = [*tokens, *(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))]
            vector = [0.0] * self.dimensions
            for feature in features:
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


# Client embedding tương thích OpenAI, dùng được với endpoint có cùng wire contract.
class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    # Nhận URL, key, model, số chiều, version và timeout; tạo client /embeddings, không trả dữ liệu.
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
        version: str,
        timeout_seconds: float,
    ) -> None:
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required for api embedding provider")
        self.model_name = model
        self.dimensions = dimensions
        self.provider_name = "openai"
        self.version = version
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
        )

    # Nhận danh sách text; gọi POST /embeddings và trả vector theo đúng index đầu vào.
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Embedding input must contain non-empty text")
        payload = {
            "input": texts,
            "model": self.model_name,
            "dimensions": self.dimensions,
            "encoding_format": "float",
        }
        try:
            response = await self.client.post("/embeddings", json=payload)
        except httpx.TransportError:
            await asyncio.sleep(0.25)
            response = await self.client.post("/embeddings", json=payload)
        response.raise_for_status()
        payload = response.json()
        ordered = sorted(payload["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in ordered]
        if len(vectors) != len(texts) or any(
            len(vector) != self.dimensions for vector in vectors
        ):
            raise ValueError("Embedding provider returned an unexpected shape")
        return vectors

    # Không nhận đầu vào; đóng connection pool HTTP và không trả dữ liệu.
    async def close(self) -> None:
        await self.client.aclose()
