from backend.embeddings import embed_text
from backend.vector_store import _collection


RETRIEVAL_TOP_K = 5


def retrieve(query, document_id, top_k=RETRIEVAL_TOP_K):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or whitespace-only.")

    if not document_id or not document_id.strip():
        raise ValueError("Document ID is required.")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    query_embedding = embed_text(query)

    result = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"document_id": document_id},
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    results = []

    for index, metadata in enumerate(metadatas):
        results.append(
            {
                "chunk_id": metadata["chunk_id"],
                "document_id": metadata["document_id"],
                "document_name": metadata["document_name"],
                "page": metadata["page"],
                "text": documents[index],
                "distance": distances[index] if index < len(distances) else None,
            }
        )

    return results