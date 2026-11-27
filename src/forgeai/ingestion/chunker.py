from pydantic import BaseModel

from .loader import Document


class Chunk(BaseModel):
    text: str
    source: str
    document_type: str
    chunk_index: int
    metadata: dict[str, str]


def chunk_document(
    document: Document,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    text = " ".join(document.text.split())
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(
            Chunk(
                text=text[start:end],
                source=document.source,
                document_type=document.document_type,
                chunk_index=index,
                metadata={
                    **document.metadata,
                    "source": document.source,
                },
            )
        )
        index += 1
        if end >= len(text):
            break
        start = end - chunk_overlap
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, chunk_overlap))
    return chunks

