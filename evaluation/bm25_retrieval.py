from backend.vector_store import _collection
from evaluation.bm25 import BM25Document, BM25Index


def build_bm25_index(document_id: str) -> BM25Index:
    if not document_id or not document_id.strip():
        raise ValueError("Document ID is required.")

    result = _collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    if len(documents) != len(metadatas):
        raise ValueError("Corpus documents and metadata counts differ.")

    corpus = [
        BM25Document(
            chunk_id=metadata["chunk_id"],
            document_id=metadata["document_id"],
            document_name=metadata["document_name"],
            page=metadata["page"],
            text=text,
        )
        for text, metadata in zip(documents, metadatas)
    ]

    corpus.sort(key=lambda item: item.chunk_id)

    return BM25Index(corpus)
