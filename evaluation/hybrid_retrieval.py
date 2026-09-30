from dataclasses import dataclass

from evaluation.bm25 import BM25Index
from evaluation.retrieval_adapter import RetrievedChunk
from evaluation.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion


@dataclass(frozen=True)
class HybridRetrievedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    text: str
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None
    rank: int


def bm25_to_retrieved_chunks(
    bm25_index: BM25Index,
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    results = bm25_index.search(query, top_k)

    return [
        RetrievedChunk(
            chunk_id=document.chunk_id,
            document_id=document.document_id,
            document_name=document.document_name,
            page=document.page,
            text=document.text,
            distance=None,
            rank=rank,
        )
        for rank, (document, _score) in enumerate(results, start=1)
    ]


def fuse_retrieval_results(
    dense_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = 5,
) -> list[HybridRetrievedChunk]:
    fused = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        k=rrf_k,
        top_k=top_k,
    )

    metadata_by_chunk_id: dict[str, RetrievedChunk] = {}

    for result in dense_results:
        metadata_by_chunk_id[result.chunk_id] = result

    for result in bm25_results:
        metadata_by_chunk_id.setdefault(result.chunk_id, result)

    output: list[HybridRetrievedChunk] = []

    for rank, result in enumerate(fused, start=1):
        metadata = metadata_by_chunk_id[result.chunk_id]

        output.append(
            HybridRetrievedChunk(
                chunk_id=result.chunk_id,
                document_id=metadata.document_id,
                document_name=metadata.document_name,
                page=metadata.page,
                text=metadata.text,
                rrf_score=result.rrf_score,
                dense_rank=result.dense_rank,
                bm25_rank=result.bm25_rank,
                rank=rank,
            )
        )

    return output