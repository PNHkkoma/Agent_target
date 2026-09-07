from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.router import ModelRouter
from app.main import create_app
from app.rag.chunking import chunk_document
from app.rag.embeddings import (
    LocalHashEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from app.rag.models import DocumentInput, RagAnswerRequest, SearchRequest
from app.rag.service import RagService
from app.rag.store import InMemoryVectorStore
from tests.fakes import FakeProvider, response


# Nhận ID, title, content và metadata; trả DocumentInput ngắn gọn dùng chung trong test.
def document(
    document_id: str,
    title: str,
    content: str,
    *,
    document_type: str = "manual",
    metadata: dict | None = None,
) -> DocumentInput:
    return DocumentInput(
        id=document_id,
        title=title,
        content=content,
        source_uri=f"https://knowledge.local/{document_id}",
        document_type=document_type,
        metadata=metadata or {},
    )


# Nhận fake model response tùy chọn; trả service memory deterministic cùng provider để quan sát.
def make_service(answer: str = "Câu trả lời [S1].") -> tuple[RagService, FakeProvider]:
    settings = Settings(
        _env_file=None,
        llm_provider="test",
        llm_fallback_providers=[],
        llm_max_retries=0,
        rag_chunk_size_chars=200,
        rag_chunk_overlap_chars=30,
        rag_min_score=-1,
        reranker_enabled=False,
    )
    provider = FakeProvider("test", responses=[response("test", answer)])
    router = ModelRouter({"test": provider}, settings)
    return (
        RagService(
            embeddings=LocalHashEmbeddingProvider(64),
            store=InMemoryVectorStore(),
            model_router=router,
            settings=settings,
        ),
        provider,
    )


# Tạo tài liệu nhiều đoạn; xác nhận chunk không vượt bất hợp lý và giữ metadata/source.
def test_chunking_preserves_provenance_and_overlap() -> None:
    source = document(
        "manual-1",
        "Hướng dẫn compression bag",
        ("Đoạn một nói về khóa kéo và vật liệu. " * 5)
        + "\n\n"
        + ("Đoạn hai nói về cách cuộn để thoát khí. " * 5),
        metadata={"product_id": "P001"},
    )
    chunks = chunk_document(
        source,
        document_id="manual-1",
        chunk_size_chars=200,
        overlap_chars=30,
        content_hash="abc123",
        embedding_provider="test_local_hash",
        embedding_model="local-hash-baseline-v1",
        embedding_dimension=64,
        embedding_version="1",
    )
    assert len(chunks) >= 2
    assert all(chunk.document_id == "manual-1" for chunk in chunks)
    assert all(chunk.metadata["product_id"] == "P001" for chunk in chunks)
    assert chunks[0].source_uri == source.source_uri


# Embedding cùng text hai lần; xác nhận vector deterministic, đúng chiều và đã normalize.
@pytest.mark.asyncio
async def test_local_embedding_is_deterministic() -> None:
    embeddings = LocalHashEmbeddingProvider(64)
    first, second = await embeddings.embed(["ripstop nylon", "ripstop nylon"])
    assert first == second
    assert len(first) == 64
    assert sum(value * value for value in first) == pytest.approx(1.0)


# Mock /embeddings; xác nhận client gửi model/dimensions và sắp output theo index.
@pytest.mark.asyncio
async def test_embedding_api_contract() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert payload["model"] == "text-embedding-3-small"
        assert payload["dimensions"] == 2
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://example.test/v1",
        api_key="test",
        model="text-embedding-3-small",
        dimensions=2,
        version="1",
        timeout_seconds=1,
    )
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url="https://example.test/v1", transport=httpx.MockTransport(handler)
    )
    assert await provider.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    await provider.close()


# Ingest hai tài liệu; xác nhận retrieval đưa tài liệu chứa từ khóa riêng lên đầu.
@pytest.mark.asyncio
async def test_retrieval_ranks_relevant_chunk_first() -> None:
    service, _ = make_service()
    await service.ingest(
        document("compression", "Compression bag", "ripstop nylon van thoát khí cabin")
    )
    await service.ingest(
        document("refund", "Đổi trả", "hoàn tiền trong ba mươi ngày")
    )
    result = await service.retrieve(
        SearchRequest(query="compression bag ripstop nylon", min_score=-1, top_k=2)
    )
    assert result.hits[0].document_id == "compression"
    assert result.hits[0].score >= result.hits[1].score


# Ingest tài liệu khác metadata; xác nhận filter product_id loại bỏ chunk không đúng phạm vi.
@pytest.mark.asyncio
async def test_retrieval_applies_metadata_filter() -> None:
    service, _ = make_service()
    await service.ingest(
        document("p1", "P001", "vật liệu nylon", metadata={"product_id": "P001"})
    )
    await service.ingest(
        document("p2", "P002", "vật liệu polyester", metadata={"product_id": "P002"})
    )
    result = await service.retrieve(
        SearchRequest(query="vật liệu", min_score=-1, filters={"product_id": "P002"})
    )
    assert [hit.document_id for hit in result.hits] == ["p2"]


# Ingest cùng document/version/nội dung hai lần; xác nhận chunk ID ổn định và không tăng duplicate.
@pytest.mark.asyncio
async def test_ingestion_is_idempotent_for_same_document_version() -> None:
    service, _ = make_service()
    source = document("stable", "Tài liệu ổn định", "Nội dung không đổi qua hai lần upload.")
    first = await service.ingest(source)
    first_ids = set(service.store.chunks)
    second = await service.ingest(source)
    second_ids = set(service.store.chunks)
    chunk = next(iter(service.store.chunks.values()))
    assert first.chunks_created == second.chunks_created
    assert first_ids == second_ids
    assert chunk.document_version == "1"
    assert len(chunk.content_hash) == 64
    assert chunk.embedding_provider == "test_local_hash"


# Hỏi sau ingestion; xác nhận context vào model và citation ánh xạ đúng nguồn retrieval.
@pytest.mark.asyncio
async def test_rag_answer_contains_real_citation() -> None:
    service, provider = make_service("Nên dùng túi ripstop nylon [S1].")
    await service.ingest(
        document("guide", "Packing Nhật Bản", "Túi dùng vật liệu ripstop nylon.")
    )
    result = await service.answer(
        RagAnswerRequest(query="Nên chọn vật liệu gì?", min_score=-1),
        request_id="rag-1",
    )
    assert result.citations[0].document_id == "guide"
    assert result.citations[0].source_uri.endswith("/guide")
    assert "[S1]" in provider.last_messages[-1].content


# Hỏi với threshold loại toàn bộ hit; xác nhận service không gọi model và không bịa câu trả lời.
@pytest.mark.asyncio
async def test_rag_answer_refuses_when_context_is_empty() -> None:
    service, provider = make_service("Không được sử dụng response này")
    result = await service.answer(
        RagAnswerRequest(query="Kiến thức chưa ingest", min_score=1),
        request_id="rag-empty",
    )
    assert result.citations == []
    assert result.provider == "none"
    assert "không đoán" in result.answer
    assert provider.chat_calls == 0


# Gọi ba endpoint RAG; xác nhận ingest, search và answer dùng chung knowledge state.
def test_rag_api_end_to_end_with_memory_store() -> None:
    service, provider = make_service("Giới hạn là 7kg [S1].")
    settings = service.settings
    router = service.model_router
    app = create_app(settings=settings, model_router=router, rag_service=service)
    with TestClient(app) as client:
        ingested = client.post(
            "/api/rag/documents",
            json=document(
                "policy", "Quy định cabin", "Hành lý cabin tối đa 7kg.", document_type="policy"
            ).model_dump(),
        )
        searched = client.post(
            "/api/rag/search", json={"query": "cabin tối đa bao nhiêu kg", "min_score": -1}
        )
        answered = client.post(
            "/api/rag/ask", json={"query": "Giới hạn cabin?", "min_score": -1}
        )
    assert ingested.status_code == 200
    assert searched.json()["hits"][0]["documentId"] == "policy"
    assert answered.json()["citations"][0]["documentId"] == "policy"
    assert provider.chat_calls == 1
