from dataclasses import dataclass

import pytest

from evaluation.rrf import reciprocal_rank_fusion


@dataclass(frozen=True)
class FakeResult:
    chunk_id: str


def result(chunk_id: str) -> FakeResult:
    return FakeResult(chunk_id=chunk_id)


def test_chunk_in_both_lists():
    fused = reciprocal_rank_fusion(
        [result("A")],
        [result("A")],
        k=60,
        top_k=5,
    )

    assert fused[0].chunk_id == "A"
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_rank == 1
    assert fused[0].rrf_score == pytest.approx(
        2 / 61
    )


def test_chunk_only_in_dense():
    fused = reciprocal_rank_fusion(
        [result("A")],
        [],
        k=60,
        top_k=5,
    )

    assert fused[0].chunk_id == "A"
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_rank is None
    assert fused[0].rrf_score == pytest.approx(1 / 61)


def test_chunk_only_in_bm25():
    fused = reciprocal_rank_fusion(
        [],
        [result("A")],
        k=60,
        top_k=5,
    )

    assert fused[0].chunk_id == "A"
    assert fused[0].dense_rank is None
    assert fused[0].bm25_rank == 1
    assert fused[0].rrf_score == pytest.approx(1 / 61)


def test_fusion_uses_actual_rrf_scores():
    # Dense: A=1, B=2
    # BM25:  B=1, A=5
    # RRF should therefore rank B above A.
    dense = [result("A"), result("B")]
    bm25 = [
        result("B"),
        result("C"),
        result("D"),
        result("E"),
        result("A"),
    ]

    fused = reciprocal_rank_fusion(
        dense,
        bm25,
        k=60,
        top_k=5,
    )

    assert [item.chunk_id for item in fused[:2]] == ["B", "A"]

    assert fused[0].rrf_score == pytest.approx(
        1 / 62 + 1 / 61
    )

    assert fused[1].rrf_score == pytest.approx(
        1 / 61 + 1 / 65
    )


def test_duplicate_chunk_is_fused_once():
    fused = reciprocal_rank_fusion(
        [result("A"), result("B")],
        [result("A"), result("B")],
        k=60,
        top_k=5,
    )

    assert [item.chunk_id for item in fused] == ["A", "B"]
    assert len(fused) == 2


def test_deterministic_tie_breaking():
    dense = [result("B"), result("A")]
    bm25 = [result("A"), result("B")]

    first = reciprocal_rank_fusion(
        dense,
        bm25,
        k=60,
        top_k=5,
    )

    second = reciprocal_rank_fusion(
        dense,
        bm25,
        k=60,
        top_k=5,
    )

    assert [
        item.chunk_id for item in first
    ] == [
        item.chunk_id for item in second
    ]

    assert [item.chunk_id for item in first] == ["A", "B"]


def test_top_k_is_enforced():
    fused = reciprocal_rank_fusion(
        [result("A"), result("B"), result("C")],
        [result("D"), result("E"), result("F")],
        k=60,
        top_k=3,
    )

    assert len(fused) == 3


def test_invalid_parameters():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [result("A")],
            [result("A")],
            k=0,
            top_k=5,
        )

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [result("A")],
            [result("A")],
            k=60,
            top_k=0,
        )
