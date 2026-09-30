from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from evaluation.hybrid_retrieval import HybridRetrievedChunk


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"
DEFAULT_FINAL_TOP_K = 5


@dataclass(frozen=True)
class RerankedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    text: str
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None
    cross_encoder_score: float
    rank: int


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model = CrossEncoder(
            model_name,
            device=device,
        )

    def rerank(
        self,
        query: str,
        candidates: list[HybridRetrievedChunk],
        *,
        top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> list[RerankedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not candidates:
            return []

        pairs = [
            (query, candidate.text)
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if len(scores) != len(candidates):
            raise RuntimeError(
                "Cross-encoder returned an unexpected number of scores."
            )

        scored = [
            (
                candidate,
                float(score),
            )
            for candidate, score in zip(candidates, scores)
        ]

        scored.sort(
            key=lambda item: (
                -item[1],
                item[0].chunk_id,
            )
        )

        return [
            RerankedChunk(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                document_name=candidate.document_name,
                page=candidate.page,
                text=candidate.text,
                rrf_score=candidate.rrf_score,
                dense_rank=candidate.dense_rank,
                bm25_rank=candidate.bm25_rank,
                cross_encoder_score=score,
                rank=rank,
            )
            for rank, (candidate, score) in enumerate(
                scored[:top_k],
                start=1,
            )
        ]
