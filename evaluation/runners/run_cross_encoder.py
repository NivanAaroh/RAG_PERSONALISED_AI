import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import sentence_transformers
import torch
import transformers

from evaluation.bm25_retrieval import build_bm25_index
from evaluation.hybrid_retrieval import (
    bm25_to_retrieved_chunks,
    fuse_retrieval_results,
)
from evaluation.retrieval_adapter import retrieve_for_evaluation
from evaluation.cross_encoder_reranker import (
    DEFAULT_DEVICE,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
)
from evaluation.metrics.retrieval import (
    evidence_recall_at_k,
    hit_at_k,
    precision_at_k,
    reciprocal_rank,
)


DENSE_CANDIDATE_K = 20
BM25_CANDIDATE_K = 20
RRF_CANDIDATE_K = 20
FINAL_TOP_K = DEFAULT_FINAL_TOP_K
RRF_K = 60


def mean(values):
    return statistics.fmean(values) if values else 0.0


def median(values):
    return statistics.median(values) if values else 0.0


def percentile(values, p):
    if not values:
        return 0.0

    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)

    if lower == upper:
        return ordered[lower]

    fraction = index - lower
    return (
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * fraction
    )


def git_revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def load_dataset(path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict) and isinstance(
        payload.get("cases"), list
    ):
        return payload["cases"]

    raise ValueError(
        "Dataset must be a list or an object containing 'cases'."
    )


def run(dataset_path, output_path):
    dataset = load_dataset(dataset_path)

    if not dataset:
        raise ValueError("Dataset is empty.")

    document_ids = {
        evidence["chunk_id"].split(":")[0]
        for case in dataset
        for evidence in case["required_evidence"]
        if isinstance(evidence, dict)
        and evidence.get("chunk_id")
    }

    if len(document_ids) != 1:
        raise ValueError(
            "Part 5 runner requires exactly one document_id."
        )

    document_id = next(iter(document_ids))

    index_start = time.perf_counter()
    bm25_index = build_bm25_index(document_id)
    index_build_ms = (
        time.perf_counter() - index_start
    ) * 1000

    model_init_start = time.perf_counter()
    reranker = CrossEncoderReranker(
        model_name=DEFAULT_RERANKER_MODEL,
        device=DEFAULT_DEVICE,
    )
    model_init_ms = (
        time.perf_counter() - model_init_start
    ) * 1000

    cases = []

    dense_latencies = []
    bm25_latencies = []
    rrf_latencies = []
    cross_encoder_latencies = []
    total_latencies = []

    supported_metrics = {
        "hit_at_5": [],
        "evidence_recall_at_5": [],
        "precision_at_5": [],
        "mrr": [],
    }

    for case in dataset:
        query = case["question"]
        support_status = case["support_status"]

        required_evidence = [
            item["chunk_id"]
            if isinstance(item, dict)
            else item
            for item in case["required_evidence"]
        ]

        total_start = time.perf_counter()

        dense_start = time.perf_counter()
        dense_results = retrieve_for_evaluation(
            query=query,
            document_id=document_id,
            top_k=DENSE_CANDIDATE_K,
        )
        dense_ms = (
            time.perf_counter() - dense_start
        ) * 1000

        bm25_start = time.perf_counter()
        bm25_results = bm25_to_retrieved_chunks(
            bm25_index,
            query,
            BM25_CANDIDATE_K,
        )
        bm25_ms = (
            time.perf_counter() - bm25_start
        ) * 1000

        rrf_start = time.perf_counter()
        hybrid_candidates = fuse_retrieval_results(
            dense_results,
            bm25_results,
            rrf_k=RRF_K,
            top_k=RRF_CANDIDATE_K,
        )
        rrf_ms = (
            time.perf_counter() - rrf_start
        ) * 1000

        cross_encoder_start = time.perf_counter()
        reranked = reranker.rerank(
            query,
            hybrid_candidates,
            top_k=FINAL_TOP_K,
        )
        cross_encoder_ms = (
            time.perf_counter() - cross_encoder_start
        ) * 1000

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000

        dense_latencies.append(dense_ms)
        bm25_latencies.append(bm25_ms)
        rrf_latencies.append(rrf_ms)
        cross_encoder_latencies.append(
            cross_encoder_ms
        )
        total_latencies.append(total_ms)

        retrieved_ids = [
            item.chunk_id
            for item in reranked
        ]

        metrics = None

        if support_status == "supported":
            metrics = {
                "hit_at_5": hit_at_k(
                    required_evidence,
                    retrieved_ids,
                    FINAL_TOP_K,
                ),
                "evidence_recall_at_5": (
                    evidence_recall_at_k(
                        required_evidence,
                        retrieved_ids,
                        FINAL_TOP_K,
                    )
                ),
                "precision_at_5": precision_at_k(
                    required_evidence,
                    retrieved_ids,
                    FINAL_TOP_K,
                ),
                "mrr": reciprocal_rank(
                    required_evidence,
                    retrieved_ids,
                ),
            }

            for key, value in metrics.items():
                supported_metrics[key].append(value)

        cases.append(
            {
                "id": case["id"],
                "question": query,
                "support_status": support_status,
                "category": case["category"],
                "required_evidence": required_evidence,
                "metrics": metrics,
                "dense_candidate_ids": [
                    item.chunk_id
                    for item in dense_results
                ],
                "bm25_candidate_ids": [
                    item.chunk_id
                    for item in bm25_results
                ],
                "rrf_candidate_ids": [
                    item.chunk_id
                    for item in hybrid_candidates
                ],
                "retrieved_chunks": [
                    {
                        "rank": item.rank,
                        "chunk_id": item.chunk_id,
                        "document_id": item.document_id,
                        "document_name": item.document_name,
                        "page": item.page,
                        "text": item.text,
                        "rrf_score": item.rrf_score,
                        "dense_rank": item.dense_rank,
                        "bm25_rank": item.bm25_rank,
                        "cross_encoder_score": (
                            item.cross_encoder_score
                        ),
                    }
                    for item in reranked
                ],
                "latency_ms": {
                    "dense": dense_ms,
                    "bm25": bm25_ms,
                    "rrf": rrf_ms,
                    "cross_encoder": cross_encoder_ms,
                    "total": total_ms,
                },
            }
        )

    output = {
        "experiment": (
            "day2_part5_cross_encoder_reranking"
        ),
        "method": "dense_bm25_rrf_cross_encoder",
        "dataset": dataset_path.name,
        "document_id": document_id,
        "corpus": {
            "source": "Chroma documents collection",
            "document_id": document_id,
            "bm25_index_documents": len(
                bm25_index.documents
            ),
        },
        "config": {
            "dense_candidate_k": DENSE_CANDIDATE_K,
            "bm25_candidate_k": BM25_CANDIDATE_K,
            "rrf_candidate_k": RRF_CANDIDATE_K,
            "final_top_k": FINAL_TOP_K,
            "rrf_k": RRF_K,
            "rrf_weighting": {
                "dense": 1.0,
                "bm25": 1.0,
            },
            "reranker_candidate_k": RRF_CANDIDATE_K,
            "reranker_model": DEFAULT_RERANKER_MODEL,
            "reranker_device": DEFAULT_DEVICE,
            "scoring_direction": "higher_is_better",
        },
        "bm25_config": {
            "k1": bm25_index.k1,
            "b": bm25_index.b,
            "tokenizer": "lowercase + [a-z0-9]+",
        },
        "reranker_config": {
            "model": DEFAULT_RERANKER_MODEL,
            "model_revision": None,
            "device": DEFAULT_DEVICE,
            "library": "sentence-transformers",
            "sentence_transformers_version": sentence_transformers.__version__,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "scoring_direction": "higher_is_better",
        },
        "index_build_latency_ms": index_build_ms,
        "model_init_latency_ms": model_init_ms,
        "aggregate": {
            "supported_cases": len(
                supported_metrics["hit_at_5"]
            ),
            "hit_at_5": mean(
                supported_metrics["hit_at_5"]
            ),
            "evidence_recall_at_5": mean(
                supported_metrics[
                    "evidence_recall_at_5"
                ]
            ),
            "precision_at_5": mean(
                supported_metrics["precision_at_5"]
            ),
            "mrr": mean(
                supported_metrics["mrr"]
            ),
        },
        "latency": {
            "dense_mean_ms": mean(dense_latencies),
            "dense_median_ms": median(dense_latencies),
            "dense_p95_ms": percentile(
                dense_latencies, 0.95
            ),
            "bm25_mean_ms": mean(bm25_latencies),
            "bm25_median_ms": median(bm25_latencies),
            "bm25_p95_ms": percentile(
                bm25_latencies, 0.95
            ),
            "rrf_mean_ms": mean(rrf_latencies),
            "rrf_median_ms": median(rrf_latencies),
            "rrf_p95_ms": percentile(
                rrf_latencies, 0.95
            ),
            "cross_encoder_mean_ms": mean(
                cross_encoder_latencies
            ),
            "cross_encoder_median_ms": median(
                cross_encoder_latencies
            ),
            "cross_encoder_p95_ms": percentile(
                cross_encoder_latencies, 0.95
            ),
            "total_mean_ms": mean(total_latencies),
            "total_median_ms": median(
                total_latencies
            ),
            "total_p95_ms": percentile(
                total_latencies, 0.95
            ),
        },
        "cases": cases,
        "git_revision": git_revision(),
        "timestamp_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            output,
            handle,
            indent=2,
        )

    print(
        f"Cross-encoder results written to: "
        f"{output_path}"
    )
    print(
        f"Supported cases: "
        f"{output['aggregate']['supported_cases']}"
    )
    print(
        f"Hit@5: "
        f"{output['aggregate']['hit_at_5']:.6f}"
    )
    print(
        "Evidence Recall@5: "
        f"{output['aggregate']['evidence_recall_at_5']:.6f}"
    )
    print(
        f"Precision@5: "
        f"{output['aggregate']['precision_at_5']:.6f}"
    )
    print(
        f"MRR: "
        f"{output['aggregate']['mrr']:.6f}"
    )
    print(
        "Latency mean ms — "
        f"dense={output['latency']['dense_mean_ms']:.3f}, "
        f"bm25={output['latency']['bm25_mean_ms']:.3f}, "
        f"rrf={output['latency']['rrf_mean_ms']:.3f}, "
        f"cross_encoder={output['latency']['cross_encoder_mean_ms']:.3f}, "
        f"total={output['latency']['total_mean_ms']:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()