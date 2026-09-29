from backend.retrieval import retrieve


def build_evidence(retrieved_chunks):
    evidence = []

    for chunk in retrieved_chunks:
        evidence.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_name": chunk["document_name"],
                "page": chunk["page"],
                "text": chunk["text"],
            }
        )

    return evidence

def has_evidence_candidates(evidence):
    return len(evidence) > 0