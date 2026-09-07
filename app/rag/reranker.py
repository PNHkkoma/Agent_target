from __future__ import annotations

import json
from dataclasses import dataclass

from app.llm.router import ModelRouter
from app.llm.errors import LLMError
from app.rag.models import SearchHit
from app.schemas.chat import ChatOptions, Message, ResponseFormat, TaskType


RERANK_SYSTEM_PROMPT = """You are a retrieval reranker. Rank only supplied chunks by their
ability to answer the query. Do not add facts. Return JSON only with this shape:
{"results":[{"chunk_id":"exact supplied id","relevance":0.0}],"confidence":0.0}.
Return only the best five chunks, each ID at most once. confidence is evidence confidence, not politeness."""


@dataclass(frozen=True)
class RerankResult:
    hits: list[SearchHit]
    confidence: float


# Reranker dùng LLM hiện có; nhận candidate và trả thứ tự relevance kèm confidence để gating.
class LLMReranker:
    # Nhận ModelRouter và giới hạn candidate; tạo reranker dùng chung provider/config ứng dụng.
    def __init__(self, model_router: ModelRouter, candidate_limit: int) -> None:
        self.model_router = model_router
        self.candidate_limit = candidate_limit

    # Nhận query, candidate và request ID; trả top candidate đã rerank, fallback an toàn nếu JSON sai.
    async def rerank(
        self, query: str, candidates: list[SearchHit], *, request_id: str
    ) -> RerankResult:
        candidates = candidates[: self.candidate_limit]
        if not candidates:
            return RerankResult(hits=[], confidence=0.0)
        payload = {
            "query": query,
            "candidates": [
                {
                    "chunk_id": hit.chunk_id,
                    "title": hit.document_title,
                    "content": hit.content[:700],
                }
                for hit in candidates
            ],
        }
        try:
            response, _ = await self.model_router.chat(
                [
                    Message(role="system", content=RERANK_SYSTEM_PROMPT),
                    Message(role="user", content=json.dumps(payload, ensure_ascii=False)),
                ],
                ChatOptions(
                    temperature=0,
                    max_tokens=200,
                    response_format=ResponseFormat.JSON_OBJECT,
                ),
                task=TaskType.DEFAULT,
                request_id=request_id,
            )
            parsed = json.loads(response.content)
            relevance = {
                item["chunk_id"]: float(item["relevance"])
                for item in parsed.get("results", [])
                if item.get("chunk_id") in {hit.chunk_id for hit in candidates}
            }
            ranked = sorted(
                candidates,
                key=lambda hit: relevance.get(hit.chunk_id, -1.0),
                reverse=True,
            )
            confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.0))))
            scored_hits = [
                SearchHit(
                    chunkId=hit.chunk_id,
                    documentId=hit.document_id,
                    documentTitle=hit.document_title,
                    sourceUri=hit.source_uri,
                    documentType=hit.document_type,
                    content=hit.content,
                    score=round(relevance.get(hit.chunk_id, 0.0), 6),
                    metadata=hit.metadata,
                )
                for hit in ranked
            ]
            return RerankResult(hits=scored_hits, confidence=confidence)
        except (LLMError, ValueError, TypeError, json.JSONDecodeError):
            return RerankResult(hits=candidates, confidence=0.0)
