from dataclasses import dataclass
from typing import Any

from backend.retrieval import retrieve


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    text: str
    distance: float | None
    rank: int


def retrieve_for_evaluation(
    query: str,
    document_id: str,
    top_k: int,
) -> list[RetrievedChunk]:
    results: list[dict[str, Any]] = retrieve(
        query=query,
        document_id=document_id,
        top_k=top_k,
    )

    return [
        RetrievedChunk(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            document_name=item["document_name"],
            page=item["page"],
            text=item["text"],
            distance=item.get("distance"),
            rank=index + 1,
        )
        for index, item in enumerate(results)
    ]
