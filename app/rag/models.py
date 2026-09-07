from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# Tài liệu do client gửi để parse, chunk, embedding và lưu vào knowledge base.
class DocumentInput(BaseModel):
    id: str | None = Field(default=None, min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=1_000_000)
    source_uri: str = Field(min_length=1, max_length=2_000)
    document_type: Literal["product", "manual", "policy", "review", "travel"]
    document_version: str = Field(default="1", min_length=1, max_length=100)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Nhận các trường text đã parse; trả document sạch khoảng trắng hoặc báo lỗi nếu rỗng.
    @field_validator("title", "content", "source_uri")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


# Một đoạn nhỏ có thể embedding, truy xuất và trích dẫn độc lập.
class DocumentChunk(BaseModel):
    id: str
    document_id: str
    document_title: str
    source_uri: str
    document_type: str
    chunk_index: int = Field(ge=0)
    document_version: str
    content_hash: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int = Field(gt=0)
    embedding_version: str
    content: str
    char_count: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list, exclude=True)


# Kết quả sau khi ingest một tài liệu vào vector store.
class IngestResponse(BaseModel):
    document_id: str = Field(alias="documentId")
    chunks_created: int = Field(alias="chunksCreated")
    embedding_model: str = Field(alias="embeddingModel")


# Query tìm kiếm kèm top-K, ngưỡng điểm và metadata filter.
class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=-1, le=1)
    filters: dict[str, Any] = Field(default_factory=dict)


# Một chunk được truy xuất cùng cosine similarity để đo chất lượng ranking.
class SearchHit(BaseModel):
    chunk_id: str = Field(alias="chunkId")
    document_id: str = Field(alias="documentId")
    document_title: str = Field(alias="documentTitle")
    source_uri: str = Field(alias="sourceUri")
    document_type: str = Field(alias="documentType")
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


# Response tìm kiếm chứa query gốc và các chunk xếp theo độ tương đồng.
class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    confidence: float | None = None
    embedding_model: str = Field(alias="embeddingModel")


# Request hỏi đáp RAG, tái sử dụng toàn bộ tùy chọn retrieval.
class RagAnswerRequest(SearchRequest):
    system_prompt: str = Field(
        default="Trả lời bằng tiếng Việt, chính xác và ngắn gọn.",
        min_length=1,
        max_length=10_000,
    )


# Citation ánh xạ câu trả lời về đúng chunk và tài liệu nguồn.
class Citation(BaseModel):
    label: str
    chunk_id: str = Field(alias="chunkId")
    document_id: str = Field(alias="documentId")
    title: str
    source_uri: str = Field(alias="sourceUri")
    excerpt: str
    score: float


# Câu trả lời RAG gồm nội dung model, nguồn và số liệu model sử dụng.
class RagAnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence_level: Literal["high", "medium", "low"] = Field(alias="confidenceLevel")
    provider: str
    model: str
    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")
    latency_ms: int = Field(alias="latencyMs")
