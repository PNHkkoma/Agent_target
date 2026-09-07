from __future__ import annotations

import re
from uuid import NAMESPACE_URL, uuid5

from app.rag.models import DocumentChunk, DocumentInput


# Nhận đoạn quá dài và kích thước tối đa; trả các mảnh cắt tại biên từ khi có thể.
def split_long_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        added = len(word) + (1 if current else 0)
        if current and current_length + added > max_chars:
            parts.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += added
    if current:
        parts.append(" ".join(current))
    return parts


# Nhận document, chunking và định danh embedding; trả chunk có nguồn, hash và ID ổn định.
def chunk_document(
    document: DocumentInput,
    *,
    document_id: str,
    chunk_size_chars: int,
    overlap_chars: int,
    content_hash: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimension: int,
    embedding_version: str,
) -> list[DocumentChunk]:
    if overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars must be smaller than chunk_size_chars")

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", document.content)
        if paragraph.strip()
    ]
    units = [
        part
        for paragraph in paragraphs
        for part in split_long_text(paragraph, chunk_size_chars)
    ]
    texts: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if current and len(candidate) > chunk_size_chars:
            texts.append(current)
            overlap = current[-overlap_chars:].lstrip() if overlap_chars else ""
            current = f"{overlap}\n\n{unit}".strip() if overlap else unit
        else:
            current = candidate
    if current:
        texts.append(current)

    return [
        DocumentChunk(
            id=str(
                uuid5(
                    NAMESPACE_URL,
                    f"{document_id}:{document.document_version}:{content_hash}:{index}",
                )
            ),
            document_id=document_id,
            document_title=document.title,
            source_uri=document.source_uri,
            document_type=document.document_type,
            chunk_index=index,
            document_version=document.document_version,
            content_hash=content_hash,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            embedding_version=embedding_version,
            content=text,
            char_count=len(text),
            metadata={**document.metadata, "document_type": document.document_type},
        )
        for index, text in enumerate(texts)
    ]
