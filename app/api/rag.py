from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.rag.models import (
    DocumentInput,
    IngestResponse,
    RagAnswerRequest,
    RagAnswerResponse,
    SearchRequest,
    SearchResponse,
)
from app.rag.service import RagService

router = APIRouter(prefix="/api/rag", tags=["rag"])


# Nhận FastAPI request; trả RagService dùng chung đã được tạo trong application state.
def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


RagDependency = Annotated[RagService, Depends(get_rag_service)]


# Nhận document và RAG service; trả ID cùng số chunk sau khi ingest thành công.
@router.post("/documents", response_model=IngestResponse, response_model_by_alias=True)
async def ingest_document(
    payload: DocumentInput, rag_service: RagDependency
) -> IngestResponse:
    return await rag_service.ingest(payload)


# Nhận query retrieval và RAG service; trả top-K chunk cùng cosine similarity.
@router.post("/search", response_model=SearchResponse, response_model_by_alias=True)
async def search_knowledge(
    payload: SearchRequest, rag_service: RagDependency
) -> SearchResponse:
    return await rag_service.retrieve(payload)


# Nhận câu hỏi, request context và RAG service; trả câu trả lời LLM kèm citations.
@router.post("/ask", response_model=RagAnswerResponse, response_model_by_alias=True)
async def ask_knowledge(
    payload: RagAnswerRequest,
    request: Request,
    rag_service: RagDependency,
) -> RagAnswerResponse:
    return await rag_service.answer(payload, request_id=request.state.request_id)
