import pytest

from evaluation.cross_encoder_reranker import (
    CrossEncoderReranker,
    RerankedChunk,
)
from evaluation.hybrid_retrieval import HybridRetrievedChunk


def candidate(
    chunk_id: str,
    *,
    rrf_score: float = 0.01,
    dense_rank: int | None = 1,
    bm25_rank: int | None = 1,
) -> HybridRetrievedChunk:
    return HybridRetrievedChunk(
        chunk_id=chunk_id,
        document_id="wp3-test",
        document_name="test.pdf",
        page=1,
        text=f"text for {chunk_id}",
        rrf_score=rrf_score,
        dense_rank=dense_rank,
        bm25_rank=bm25_rank,
        rank=1,
    )


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.inputs = []

    def predict(self, pairs, **kwargs):
        self.inputs.append(pairs)
        return self.scores


def reranker_with_scores(scores):
    reranker = object.__new__(CrossEncoderReranker)
    reranker.model_name = "fake-model"
    reranker.device = "cpu"
    reranker.model = FakeCrossEncoder(scores)
    return reranker


def test_query_candidate_pairing():
    reranker = reranker_with_scores([0.5, 0.4])
    candidates = [candidate("A"), candidate("B")]

    reranker.rerank("What is RAG?", candidates, top_k=5)

    assert reranker.model.inputs == [[
        ("What is RAG?", "text for A"),
        ("What is RAG?", "text for B"),
    ]]


def test_each_candidate_gets_exactly_one_score():
    reranker = reranker_with_scores([0.9, 0.2, 0.7])
    candidates = [candidate("A"), candidate("B"), candidate("C")]

    results = reranker.rerank("query", candidates, top_k=5)

    assert len(results) == 3
    assert [item.cross_encoder_score for item in results] == [0.9, 0.7, 0.2]


def test_higher_score_ranks_first():
    reranker = reranker_with_scores([0.2, 0.9, 0.5])
    candidates = [candidate("A"), candidate("B"), candidate("C")]

    results = reranker.rerank("query", candidates, top_k=5)

    assert [item.chunk_id for item in results] == ["B", "C", "A"]


def test_ties_are_deterministic_by_chunk_id():
    reranker = reranker_with_scores([0.5, 0.5, 0.5])
    candidates = [candidate("C"), candidate("A"), candidate("B")]

    results = reranker.rerank("query", candidates, top_k=5)

    assert [item.chunk_id for item in results] == ["A", "B", "C"]


def test_candidate_preservation_before_final_truncation():
    reranker = reranker_with_scores([0.1, 0.2, 0.3, 0.4])
    candidates = [
        candidate("A"),
        candidate("B"),
        candidate("C"),
        candidate("D"),
    ]

    results = reranker.rerank("query", candidates, top_k=3)

    assert {item.chunk_id for item in results}.issubset(
        {"A", "B", "C", "D"}
    )
    assert len(results) == 3


def test_metadata_preserved():
    original = HybridRetrievedChunk(
        chunk_id="A",
        document_id="wp3-test",
        document_name="source.pdf",
        page=7,
        text="important evidence",
        rrf_score=0.02,
        dense_rank=3,
        bm25_rank=5,
        rank=2,
    )

    reranker = reranker_with_scores([1.25])
    result = reranker.rerank("query", [original], top_k=5)[0]

    assert result.chunk_id == "A"
    assert result.document_id == "wp3-test"
    assert result.document_name == "source.pdf"
    assert result.page == 7
    assert result.text == "important evidence"
    assert result.rrf_score == 0.02
    assert result.dense_rank == 3
    assert result.bm25_rank == 5
    assert result.cross_encoder_score == 1.25
    assert result.rank == 1


def test_final_top_k_is_enforced():
    reranker = reranker_with_scores([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    candidates = [
        candidate("A"),
        candidate("B"),
        candidate("C"),
        candidate("D"),
        candidate("E"),
        candidate("F"),
    ]

    results = reranker.rerank("query", candidates, top_k=5)

    assert len(results) == 5


def test_output_cannot_contain_absent_candidate():
    reranker = reranker_with_scores([0.1, 0.9])
    candidates = [candidate("A"), candidate("B")]

    results = reranker.rerank("query", candidates, top_k=5)

    assert {item.chunk_id for item in results} == {"A", "B"}


def test_empty_candidates_are_deterministically_empty():
    reranker = reranker_with_scores([])

    assert reranker.rerank("query", [], top_k=5) == []


def test_repeatability():
    candidates = [candidate("C"), candidate("A"), candidate("B")]

    first = reranker_with_scores([0.7, 0.7, 0.2]).rerank(
        "query",
        candidates,
        top_k=3,
    )
    second = reranker_with_scores([0.7, 0.7, 0.2]).rerank(
        "query",
        candidates,
        top_k=3,
    )

    assert [
        (item.chunk_id, item.cross_encoder_score, item.rank)
        for item in first
    ] == [
        (item.chunk_id, item.cross_encoder_score, item.rank)
        for item in second
    ]
