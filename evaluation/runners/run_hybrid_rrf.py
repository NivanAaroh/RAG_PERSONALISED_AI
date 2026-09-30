import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation.bm25_retrieval import build_bm25_index
from evaluation.hybrid_retrieval import (
    bm25_to_retrieved_chunks,
    fuse_retrieval_results,
)
from evaluation.metrics.retrieval import (
    evidence_recall_at_k,
    hit_at_k,
    precision_at_k,
    reciprocal_rank,
)
from evaluation.retrieval_adapter import retrieve_for_evaluation


DENSE_CANDIDATE_K = 20
BM25_CANDIDATE_K = 20
FINAL_TOP_K = 5
RRF_K = 60


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)

    if lower == upper:
        return ordered[lower]

    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return payload["cases"]

    raise ValueError("Dataset must be a list or an object containing 'cases'.")


def run(dataset_path: Path, output_path: Path) -> None:
    dataset = load_dataset(dataset_path)

    if not dataset:
        raise ValueError("Dataset is empty.")

    document_ids = {
        evidence["chunk_id"].split(":")[0]
        for case in dataset
        for evidence in case["required_evidence"]
        if isinstance(evidence, dict) and evidence.get("chunk_id")
    }

    if len(document_ids) != 1:
        raise ValueError(
            "Hybrid runner requires exactly one document_id in the benchmark."
        )

    document_id = next(iter(document_ids))

    # Build the sparse index once. Index construction is excluded from query latency.
    index_start = time.perf_counter()
    bm25_index = build_bm25_index(document_id)
    index_build_ms = (time.perf_counter() - index_start) * 1000

    cases = []

    dense_latencies = []
    bm25_latencies = []
    rrf_latencies = []
    total_latencies = []

    supported_metrics = {
        "hit_at_5": [],
        "evidence_recall_at_5": [],
        "precision_at_5": [],
        "mrr": [],
    }

    for case in dataset:
        query = case["question"]

        raw_required_evidence = case["required_evidence"]
        support_status = case["support_status"]

        required_evidence = [
            item["chunk_id"] if isinstance(item, dict) else item
            for item in raw_required_evidence
        ]

        total_start = time.perf_counter()

        dense_start = time.perf_counter()
        dense_results = retrieve_for_evaluation(
            query=query,
            document_id=document_id,
            top_k=DENSE_CANDIDATE_K,
        )
        dense_ms = (time.perf_counter() - dense_start) * 1000

        bm25_start = time.perf_counter()
        bm25_results = bm25_to_retrieved_chunks(
            bm25_index,
            query,
            BM25_CANDIDATE_K,
        )
        bm25_ms = (time.perf_counter() - bm25_start) * 1000

        rrf_start = time.perf_counter()
        hybrid_results = fuse_retrieval_results(
            dense_results,
            bm25_results,
            rrf_k=RRF_K,
            top_k=FINAL_TOP_K,
        )
        rrf_ms = (time.perf_counter() - rrf_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        dense_latencies.append(dense_ms)
        bm25_latencies.append(bm25_ms)
        rrf_latencies.append(rrf_ms)
        total_latencies.append(total_ms)

        retrieved_ids = [item.chunk_id for item in hybrid_results]

        metrics = None

        if support_status == "supported":
            metrics = {
                "hit_at_5": hit_at_k(
                    required_evidence,
                    retrieved_ids,
                    FINAL_TOP_K,
                ),
                "evidence_recall_at_5": evidence_recall_at_k(
                    required_evidence,
                    retrieved_ids,
                    FINAL_TOP_K,
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
                    }
                    for item in hybrid_results
                ],
                "latency_ms": {
                    "dense": dense_ms,
                    "bm25": bm25_ms,
                    "rrf": rrf_ms,
                    "total": total_ms,
                },
            }
        )

    output = {
        "experiment": "day2_part4_hybrid_rrf",
        "method": "dense_bm25_rrf",
        "dataset": dataset_path.name,
        "document_id": document_id,
        "corpus": {
            "source": "Chroma documents collection",
            "document_id": document_id,
            "bm25_index_documents": len(bm25_index.documents),
        },
        "config": {
            "dense_candidate_k": DENSE_CANDIDATE_K,
            "bm25_candidate_k": BM25_CANDIDATE_K,
            "final_top_k": FINAL_TOP_K,
            "rrf_k": RRF_K,
            "dense_weight": 1.0,
            "bm25_weight": 1.0,
        },
        "bm25_config": {
            "k1": bm25_index.k1,
            "b": bm25_index.b,
            "tokenizer": "lowercase + [a-z0-9]+",
        },
        "index_build_latency_ms": index_build_ms,
        "aggregate": {
            "supported_cases": len(supported_metrics["hit_at_5"]),
            "hit_at_5": mean(supported_metrics["hit_at_5"]),
            "evidence_recall_at_5": mean(
                supported_metrics["evidence_recall_at_5"]
            ),
            "precision_at_5": mean(supported_metrics["precision_at_5"]),
            "mrr": mean(supported_metrics["mrr"]),
        },
        "latency": {
            "dense_mean_ms": mean(dense_latencies),
            "dense_median_ms": median(dense_latencies),
            "dense_p95_ms": percentile(dense_latencies, 0.95),
            "bm25_mean_ms": mean(bm25_latencies),
            "bm25_median_ms": median(bm25_latencies),
            "bm25_p95_ms": percentile(bm25_latencies, 0.95),
            "rrf_mean_ms": mean(rrf_latencies),
            "rrf_median_ms": median(rrf_latencies),
            "rrf_p95_ms": percentile(rrf_latencies, 0.95),
            "total_mean_ms": mean(total_latencies),
            "total_median_ms": median(total_latencies),
            "total_p95_ms": percentile(total_latencies, 0.95),
        },
        "cases": cases,
        "git_revision": git_revision(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Hybrid RRF results written to: {output_path}")
    print(f"Supported cases: {output['aggregate']['supported_cases']}")
    print(f"Hit@5: {output['aggregate']['hit_at_5']:.6f}")
    print(
        "Evidence Recall@5: "
        f"{output['aggregate']['evidence_recall_at_5']:.6f}"
    )
    print(f"Precision@5: {output['aggregate']['precision_at_5']:.6f}")
    print(f"MRR: {output['aggregate']['mrr']:.6f}")
    print(
        "Latency mean ms — "
        f"dense={output['latency']['dense_mean_ms']:.3f}, "
        f"bm25={output['latency']['bm25_mean_ms']:.3f}, "
        f"rrf={output['latency']['rrf_mean_ms']:.3f}, "
        f"total={output['latency']['total_mean_ms']:.3f}"
    )


def main() -> None:
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