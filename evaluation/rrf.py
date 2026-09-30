from dataclasses import dataclass


DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class RRFResult:
    chunk_id: str
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None


def reciprocal_rank_fusion(
    dense_results: list,
    bm25_results: list,
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int = 5,
) -> list[RRFResult]:
    if k <= 0:
        raise ValueError("RRF k must be positive.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result.chunk_id
        dense_ranks[chunk_id] = rank
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (
            1.0 / (k + rank)
        )

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result.chunk_id
        bm25_ranks[chunk_id] = rank
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (
            1.0 / (k + rank)
        )

    fused = [
        RRFResult(
            chunk_id=chunk_id,
            rrf_score=score,
            dense_rank=dense_ranks.get(chunk_id),
            bm25_rank=bm25_ranks.get(chunk_id),
        )
        for chunk_id, score in scores.items()
    ]

    fused.sort(
        key=lambda item: (-item.rrf_score, item.chunk_id)
    )

    return fused[:top_k]
