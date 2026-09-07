from __future__ import annotations

import asyncio

from app.rag.chunking import chunk_document
from app.rag.embeddings import LocalHashEmbeddingProvider
from app.rag.models import DocumentInput
from app.rag.pgvector_store import PgVectorStore


# Không nhận đầu vào; tạo schema, upsert một tài liệu và xác nhận cosine search trên pgvector.
async def main() -> None:
    embeddings = LocalHashEmbeddingProvider(384)
    store = PgVectorStore(
        "postgresql://agent:agent@localhost:5433/agent_lab",
        dimensions=384,
        embedding_provider=embeddings.provider_name,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.version,
    )
    await store.initialize()
    document = DocumentInput(
        id="pgvector-smoke",
        title="Quy định hành lý cabin",
        content="Hành lý cabin trong bộ dữ liệu thử nghiệm có giới hạn 7kg.",
        source_uri="https://knowledge.local/policies/cabin",
        document_type="policy",
        metadata={"market": "demo"},
    )
    chunks = chunk_document(
        document,
        document_id=document.id,
        chunk_size_chars=800,
        overlap_chars=120,
        content_hash="pgvector-smoke-v1",
        embedding_provider=embeddings.provider_name,
        embedding_model=embeddings.model_name,
        embedding_dimension=embeddings.dimensions,
        embedding_version=embeddings.version,
    )
    vectors = await embeddings.embed([chunk.content for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector
    await store.upsert_document(document, document.id, chunks)
    query_vector = (await embeddings.embed(["giới hạn hành lý cabin 7kg"]))[0]
    hits = await store.search(
        query_vector,
        top_k=3,
        min_score=-1,
        filters={"market": "demo"},
    )
    if not hits or hits[0].document_id != document.id:
        raise SystemExit("pgvector smoke failed: expected document was not retrieved")
    print(
        f"pgvector smoke passed: document={hits[0].document_id} score={hits[0].score}"
    )


if __name__ == "__main__":
    asyncio.run(main())
