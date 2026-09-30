from dataclasses import dataclass

from evaluation.hybrid_retrieval import fuse_retrieval_results
from evaluation.retrieval_adapter import RetrievedChunk


@dataclass(frozen=True)
class FakeResult:
    chunk_id: str
    document_id: str = "doc-1"
    document_name: str = "test.pdf"
    page: int = 1
    text: str = "test text"
    distance: float | None = None
    rank: int = 1


def make_result(chunk_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_name="test.pdf",
        page=rank,
        text=f"text for {chunk_id}",
        distance=None,
        rank=rank,
    )


def test_hybrid_preserves_metadata():
    dense = [
        make_result("A", 1),
        make_result("B", 2),
    ]
    bm25 = [
        make_result("B", 1),
        make_result("C", 2),
    ]

    results = fuse_retrieval_results(
        dense,
        bm25,
        rrf_k=60,
        top_k=3,
    )

    assert len(results) == 3

    result_a = next(item for item in results if item.chunk_id == "A")

    assert result_a.document_id == "doc-1"
    assert result_a.document_name == "test.pdf"
    assert result_a.text == "text for A"
    assert result_a.dense_rank == 1
    assert result_a.bm25_rank is None


def test_hybrid_preserves_both_ranks():
    dense = [
        make_result("A", 1),
        make_result("B", 2),
    ]
    bm25 = [
        make_result("B", 1),
        make_result("A", 2),
    ]

    results = fuse_retrieval_results(
        dense,
        bm25,
        rrf_k=60,
        top_k=2,
    )

    result_a = next(item for item in results if item.chunk_id == "A")
    result_b = next(item for item in results if item.chunk_id == "B")

    assert result_a.dense_rank == 1
    assert result_a.bm25_rank == 2

    assert result_b.dense_rank == 2
    assert result_b.bm25_rank == 1


def test_hybrid_has_unique_chunks():
    dense = [
        make_result("A", 1),
        make_result("B", 2),
    ]
    bm25 = [
        make_result("A", 1),
        make_result("B", 2),
    ]

    results = fuse_retrieval_results(
        dense,
        bm25,
        rrf_k=60,
        top_k=5,
    )

    chunk_ids = [item.chunk_id for item in results]

    assert chunk_ids == ["A", "B"]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_hybrid_top_k_is_enforced():
    dense = [
        make_result("A", 1),
        make_result("B", 2),
        make_result("C", 3),
    ]
    bm25 = [
        make_result("D", 1),
        make_result("E", 2),
        make_result("F", 3),
    ]

    results = fuse_retrieval_results(
        dense,
        bm25,
        rrf_k=60,
        top_k=3,
    )

    assert len(results) == 3
    assert [item.rank for item in results] == [1, 2, 3]