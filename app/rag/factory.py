from app.config import Settings
from app.llm.router import ModelRouter
from app.rag.embeddings import OpenAICompatibleEmbeddingProvider
from app.rag.pgvector_store import PgVectorStore
from app.rag.service import RagService
from app.rag.store import InMemoryVectorStore


# Nhận settings và model router; trả RagService dùng backend embedding/store theo cấu hình.
def build_rag_service(settings: Settings, model_router: ModelRouter) -> RagService:
    if settings.embedding_provider in {"openai", "api"}:
        embeddings = OpenAICompatibleEmbeddingProvider(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key or settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            version=settings.embedding_version,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")

    if settings.rag_store == "memory":
        store = InMemoryVectorStore(
            embedding_provider=embeddings.provider_name,
            embedding_model=embeddings.model_name,
            embedding_dimensions=embeddings.dimensions,
            embedding_version=embeddings.version,
        )
    elif settings.rag_store == "postgres":
        store = PgVectorStore(
            settings.rag_database_url,
            settings.embedding_dimensions,
            embedding_provider=embeddings.provider_name,
            embedding_model=embeddings.model_name,
            embedding_version=embeddings.version,
            hnsw_iterative_scan=settings.rag_hnsw_iterative_scan,
        )
    else:
        raise ValueError(f"Unsupported RAG_STORE: {settings.rag_store}")
    return RagService(
        embeddings=embeddings,
        store=store,
        model_router=model_router,
        settings=settings,
    )
