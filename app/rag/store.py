from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from app.rag.models import DocumentChunk, DocumentInput, SearchHit


# Contract lưu và tìm chunk, cho phép đổi memory sang pgvector mà service không đổi.
class VectorStore(ABC):
    # Không nhận đầu vào; chuẩn bị schema/tài nguyên và không trả dữ liệu.
    @abstractmethod
    async def initialize(self) -> None:
        raise NotImplementedError

    # Nhận document cùng chunk đã embedding; thay thế dữ liệu cũ và trả số chunk lưu được.
    @abstractmethod
    async def upsert_document(
        self, document: DocumentInput, document_id: str, chunks: list[DocumentChunk]
    ) -> int:
        raise NotImplementedError

    # Nhận query vector, top-K, threshold và filter; trả các hit xếp theo cosine similarity.
    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        min_score: float,
        filters: dict[str, Any],
    ) -> list[SearchHit]:
        raise NotImplementedError

    # Nhận query text, top-K và filter; trả keyword hit để ghép với semantic candidate.
    @abstractmethod
    async def lexical_search(
        self, query: str, *, top_k: int, filters: dict[str, Any]
    ) -> list[SearchHit]:
        raise NotImplementedError

    # Không nhận đầu vào; đóng kết nối store và không trả dữ liệu.
    async def close(self) -> None:
        return None


# Vector store trong RAM dùng cho unit test và baseline không cần PostgreSQL.
class InMemoryVectorStore(VectorStore):
    # Nhận định danh không gian embedding tùy chọn; tạo memory store chỉ search các vector cùng loại.
    def __init__(
        self,
        *,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
        embedding_version: str | None = None,
    ) -> None:
        self.documents: dict[str, DocumentInput] = {}
        self.chunks: dict[str, DocumentChunk] = {}
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.embedding_version = embedding_version

    # Không nhận đầu vào; memory store không cần tạo schema nên kết thúc ngay.
    async def initialize(self) -> None:
        return None

    # Nhận document và chunk; xóa phiên bản cùng ID, lưu bản mới và trả số chunk.
    async def upsert_document(
        self, document: DocumentInput, document_id: str, chunks: list[DocumentChunk]
    ) -> int:
        self.documents[document_id] = document
        self.chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self.chunks.items()
            if chunk.document_id != document_id
        }
        self.chunks.update({chunk.id: chunk for chunk in chunks})
        return len(chunks)

    # Nhận hai vector; trả cosine similarity, trong đó 1 là cùng hướng và -1 là đối nghịch.
    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(
            sum(y * y for y in right)
        )
        return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0

    # Nhận metadata và filter; trả true khi mọi cặp filter đều khớp chính xác.
    @staticmethod
    def _matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        return all(metadata.get(key) == value for key, value in filters.items())

    # Nhận query vector và điều kiện; trả top-K hit đã filter và sắp điểm giảm dần.
    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        min_score: float,
        filters: dict[str, Any],
    ) -> list[SearchHit]:
        ranked = [
            (self._cosine(query_embedding, chunk.embedding), chunk)
            for chunk in self.chunks.values()
            if self._matches(chunk.metadata, filters)
            and (
                self.embedding_provider is None
                or (
                    chunk.embedding_provider == self.embedding_provider
                    and chunk.embedding_model == self.embedding_model
                    and chunk.embedding_dimension == self.embedding_dimensions
                    and chunk.embedding_version == self.embedding_version
                )
            )
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchHit(
                chunkId=chunk.id,
                documentId=chunk.document_id,
                documentTitle=chunk.document_title,
                sourceUri=chunk.source_uri,
                documentType=chunk.document_type,
                content=chunk.content,
                score=round(score, 6),
                metadata=chunk.metadata,
            )
            for score, chunk in ranked[:top_k]
            if score >= min_score
        ]

    # Nhận query text; trả lexical ranking đơn giản cho test khi không có PostgreSQL FTS.
    async def lexical_search(
        self, query: str, *, top_k: int, filters: dict[str, Any]
    ) -> list[SearchHit]:
        query_terms = {term.casefold() for term in query.replace("-", " ").split()}
        ranked: list[tuple[float, DocumentChunk]] = []
        for chunk in self.chunks.values():
            if not self._matches(chunk.metadata, filters):
                continue
            content_terms = set(
                (chunk.document_title + " " + chunk.content).casefold().replace("-", " ").split()
            )
            score = float(len(query_terms & content_terms))
            if score:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchHit(
                chunkId=chunk.id,
                documentId=chunk.document_id,
                documentTitle=chunk.document_title,
                sourceUri=chunk.source_uri,
                documentType=chunk.document_type,
                content=chunk.content,
                score=score,
                metadata=chunk.metadata,
            )
            for score, chunk in ranked[:top_k]
        ]
