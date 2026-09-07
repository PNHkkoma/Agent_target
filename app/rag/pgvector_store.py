from __future__ import annotations

import asyncio
from typing import Any, Literal

from app.rag.models import DocumentChunk, DocumentInput, SearchHit
from app.rag.store import VectorStore


# PostgreSQL/pgvector store dùng cosine distance và metadata JSONB filter.
class PgVectorStore(VectorStore):
    # Nhận DSN cùng model/dimension/version; lưu cấu hình để chỉ search đúng không gian vector.
    def __init__(
        self,
        database_url: str,
        dimensions: int,
        *,
        embedding_provider: str = "unknown",
        embedding_model: str = "unknown",
        embedding_version: str = "1",
        hnsw_iterative_scan: Literal["off", "strict_order", "relaxed_order"] = "strict_order",
    ) -> None:
        self.database_url = database_url
        self.dimensions = dimensions
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_version = embedding_version
        if hnsw_iterative_scan not in {"off", "strict_order", "relaxed_order"}:
            raise ValueError("Unsupported hnsw iterative scan mode")
        self.hnsw_iterative_scan = hnsw_iterative_scan
        self.table_name = "rag_chunks" if dimensions == 384 else f"rag_chunks_{dimensions}"

    # Không nhận đầu vào; mở sync connection đã bật extension/register vector và trả connection.
    def _connect(self):
        from pgvector.psycopg import register_vector
        from psycopg import connect

        connection = connect(self.database_url)
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)
        return connection

    # Không nhận đầu vào; tạo extension, bảng, metadata index và HNSW cosine index đồng bộ.
    def _initialize_sync(self) -> None:
        if not 8 <= self.dimensions <= 2_000:
            raise ValueError("pgvector dimensions must be between 8 and 2000")
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    document_version TEXT NOT NULL DEFAULT '1',
                    content_hash TEXT NOT NULL DEFAULT '',
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    content TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    document_version TEXT NOT NULL DEFAULT '1',
                    content_hash TEXT NOT NULL DEFAULT '',
                    embedding_provider TEXT NOT NULL DEFAULT 'unknown',
                    embedding_model TEXT NOT NULL DEFAULT 'unknown',
                    embedding_dimension INTEGER NOT NULL DEFAULT {self.dimensions},
                    embedding_version TEXT NOT NULL DEFAULT '1',
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
                    embedding vector({self.dimensions}) NOT NULL,
                    UNIQUE(document_id, chunk_index)
                )
                """
            )
            connection.execute(
                "ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS document_version TEXT NOT NULL DEFAULT '1'"
            )
            connection.execute(
                "ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT ''"
            )
            for statement in (
                f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS document_version TEXT NOT NULL DEFAULT '1'",
                f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT ''",
                f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS embedding_provider TEXT NOT NULL DEFAULT 'unknown'",
                f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS embedding_model TEXT NOT NULL DEFAULT 'unknown'",
                f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS embedding_dimension INTEGER NOT NULL DEFAULT {self.dimensions}",
                f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS embedding_version TEXT NOT NULL DEFAULT '1'",
                f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS search_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED",
            ):
                connection.execute(statement)
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_idx ON {self.table_name} USING gin (metadata)"
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw_idx "
                f"ON {self.table_name} USING hnsw (embedding vector_cosine_ops)"
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table_name}_fts_idx ON {self.table_name} USING gin (search_vector)"
            )

    # Không nhận đầu vào; chạy khởi tạo SQL trong worker thread và không trả dữ liệu.
    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    # Nhận document/chunks; upsert và thay toàn bộ chunk trong một transaction đồng bộ.
    def _upsert_document_sync(
        self, document: DocumentInput, document_id: str, chunks: list[DocumentChunk]
    ) -> int:
        from pgvector import Vector
        from psycopg.types.json import Jsonb

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rag_documents
                    (id, title, source_uri, document_type, document_version, content_hash, metadata, content)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    source_uri = EXCLUDED.source_uri,
                    document_type = EXCLUDED.document_type,
                    document_version = EXCLUDED.document_version,
                    content_hash = EXCLUDED.content_hash,
                    metadata = EXCLUDED.metadata,
                    content = EXCLUDED.content,
                    updated_at = NOW()
                """,
                (
                    document_id,
                    document.title,
                    document.source_uri,
                    document.document_type,
                    chunks[0].document_version,
                    chunks[0].content_hash,
                    Jsonb(document.metadata),
                    document.content,
                ),
            )
            connection.execute(
                f"DELETE FROM {self.table_name} WHERE document_id = %s", (document_id,)
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {self.table_name}
                        (id, document_id, chunk_index, content, char_count,
                         document_version, content_hash, embedding_provider, embedding_model,
                         embedding_dimension, embedding_version, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            chunk.id,
                            document_id,
                            chunk.chunk_index,
                            chunk.content,
                            chunk.char_count,
                            chunk.document_version,
                            chunk.content_hash,
                            chunk.embedding_provider,
                            chunk.embedding_model,
                            chunk.embedding_dimension,
                            chunk.embedding_version,
                            Jsonb(chunk.metadata),
                            Vector(chunk.embedding),
                        )
                        for chunk in chunks
                    ],
                )
        return len(chunks)

    # Nhận document/chunks; chạy transaction trong worker thread và trả số chunk đã lưu.
    async def upsert_document(
        self, document: DocumentInput, document_id: str, chunks: list[DocumentChunk]
    ) -> int:
        return await asyncio.to_thread(
            self._upsert_document_sync, document, document_id, chunks
        )

    # Nhận query vector và điều kiện; chạy cosine SQL đồng bộ rồi trả SearchHit đã xếp hạng.
    def _search_sync(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        min_score: float,
        filters: dict[str, Any],
    ) -> list[SearchHit]:
        from pgvector import Vector
        from psycopg.types.json import Jsonb

        filter_sql = "AND c.metadata @> %s" if filters else ""
        query = f"""
            SELECT c.id, c.document_id, d.title, d.source_uri, d.document_type,
                   c.content, c.metadata, 1 - (c.embedding <=> %s) AS score
            FROM {self.table_name} c
            JOIN rag_documents d ON d.id = c.document_id
            WHERE c.embedding_provider = %s
              AND c.embedding_model = %s
              AND c.embedding_dimension = %s
              AND c.embedding_version = %s
              {filter_sql}
              AND 1 - (c.embedding <=> %s) >= %s
            ORDER BY c.embedding <=> %s
            LIMIT %s
        """
        vector = Vector(query_embedding)
        parameters: list[Any] = [
            vector,
            self.embedding_provider,
            self.embedding_model,
            self.dimensions,
            self.embedding_version,
        ]
        if filters:
            parameters.append(Jsonb(filters))
        parameters.extend([vector, min_score, vector, top_k])
        with self._connect() as connection:
            if filters and self.hnsw_iterative_scan:
                connection.execute(
                    f"SET LOCAL hnsw.iterative_scan = '{self.hnsw_iterative_scan}'"
                )
            rows = connection.execute(query, parameters).fetchall()
        return [
            SearchHit(
                chunkId=row[0],
                documentId=row[1],
                documentTitle=row[2],
                sourceUri=row[3],
                documentType=row[4],
                content=row[5],
                metadata=row[6],
                score=round(float(row[7]), 6),
            )
            for row in rows
        ]

    # Nhận query vector và điều kiện; chạy search trong worker thread rồi trả top-K hit.
    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        min_score: float,
        filters: dict[str, Any],
    ) -> list[SearchHit]:
        return await asyncio.to_thread(
            self._search_sync,
            query_embedding,
            top_k=top_k,
            min_score=min_score,
            filters=filters,
        )

    # Nhận query text và filter; chạy PostgreSQL FTS simple dictionary, trả candidate keyword xếp hạng.
    def _lexical_search_sync(
        self, query_text: str, *, top_k: int, filters: dict[str, Any]
    ) -> list[SearchHit]:
        from psycopg.types.json import Jsonb

        filter_sql = "AND c.metadata @> %s" if filters else ""
        query = f"""
            WITH search_query AS (SELECT websearch_to_tsquery('simple', %s) AS value)
            SELECT c.id, c.document_id, d.title, d.source_uri, d.document_type,
                   c.content, c.metadata,
                   ts_rank_cd(c.search_vector, search_query.value) AS score
            FROM {self.table_name} c
            JOIN rag_documents d ON d.id = c.document_id
            CROSS JOIN search_query
            WHERE c.embedding_provider = %s
              AND c.embedding_model = %s
              AND c.embedding_dimension = %s
              AND c.embedding_version = %s
              AND c.search_vector @@ search_query.value
              {filter_sql}
            ORDER BY score DESC, c.id
            LIMIT %s
        """
        parameters: list[Any] = [
            query_text,
            self.embedding_provider,
            self.embedding_model,
            self.dimensions,
            self.embedding_version,
        ]
        if filters:
            parameters.append(Jsonb(filters))
        parameters.append(top_k)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            SearchHit(
                chunkId=row[0], documentId=row[1], documentTitle=row[2],
                sourceUri=row[3], documentType=row[4], content=row[5],
                metadata=row[6], score=round(float(row[7]), 6),
            )
            for row in rows
        ]

    # Nhận query và filter; chạy FTS worker thread rồi trả keyword candidate cho hybrid merge.
    async def lexical_search(
        self, query: str, *, top_k: int, filters: dict[str, Any]
    ) -> list[SearchHit]:
        return await asyncio.to_thread(
            self._lexical_search_sync, query, top_k=top_k, filters=filters
        )
