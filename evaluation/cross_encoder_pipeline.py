from evaluation.bm25_retrieval import build_bm25_index
from evaluation.hybrid_retrieval import (
    bm25_to_retrieved_chunks,
    fuse_retrieval_results,
)
from evaluation.retrieval_adapter import retrieve_for_evaluation


DENSE_CANDIDATE_K = 20
BM25_CANDIDATE_K = 20
RRF_CANDIDATE_K = 20


def build_hybrid_candidate_pool(
    query: str,
    document_id: str,
    bm25_index,
):
    dense_results = retrieve_for_evaluation(
        query=query,
        document_id=document_id,
        top_k=DENSE_CANDIDATE_K,
    )

    bm25_results = bm25_to_retrieved_chunks(
        bm25_index,
        query,
        BM25_CANDIDATE_K,
    )

    hybrid_candidates = fuse_retrieval_results(
        dense_results,
        bm25_results,
        rrf_k=60,
        top_k=RRF_CANDIDATE_K,
    )

    return dense_results, bm25_results, hybrid_candidates
